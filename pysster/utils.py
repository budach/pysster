import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gzip
import os
import pickle
from itertools import groupby, repeat
from subprocess import check_output, call
from os.path import dirname
import forgi.graph.bulge_graph as cgb
import numpy as np
from shutil import which, move
from collections import Counter
from os import remove
from tempfile import gettempdir
from math import ceil


def save_model(model, file_path):
    """ Save a pysster.Model object.

    This function creates two files: a pickled version of the pysster.Model object and
    an hdf5 file of the actual keras model (e.g. if file_path is 'model' two files are
    created: 'model' and 'model.h5')

    Parameters
    ----------
    model : pysster.Model
        A Model object.
    
    file_path : str
        A file name.
    """
    with gzip.open(file_path, "wb") as handle:
        pickle.dump(model.params, handle, pickle.HIGHEST_PROTOCOL)
    model.model.save("{}.h5".format(file_path))


def load_model(file_path):
    """ Load a pysster.Model object.

    Parameters
    ----------
    file_path : str
       A file containing a pickled pysster.Model object (file_path.h5 must also exist, see save_model()).

    Returns
    -------
    model : pysster.Model
        A Model object.
    """
    from pysster.Model import Model
    from keras.models import load_model as load_keras
    if not os.path.exists(file_path):
        raise RuntimeError("Path not found.")
    if not os.path.exists("{}.h5".format(file_path)):
        raise RuntimeError("HDF5 file not found.")
    with gzip.open(file_path, "rb") as handle:
        params = pickle.load(handle)
    model = Model(params, None)
    model.model = load_keras("{}.h5".format(file_path))
    return model


def save_data(data, file_path):
    """ Save a pysster.Data object.

    The object will be pickled to disk.

    Parameters
    ----------
    file_path : str
        A file name.
    """
    with gzip.open(file_path, "wb") as handle:
        pickle.dump(data, handle, pickle.HIGHEST_PROTOCOL)


def load_data(file_path):
    """ Load a pysster.Data object.

    Parameters
    ----------
    file_path : str
        A file containing a pickled pysster.Data object.
    
    Returns
    -------
    data : pysster.Data
        The Data object loaded from file.
    """
    with gzip.open(file_path, "rb") as handle:
        return pickle.load(handle)


def get_handle(file_name, mode):
    if file_name[-2:] == "gz":
        return gzip.open(file_name, mode)
    return open(file_name, mode)


def parse_fasta(handle, joiner = ""):
    delimiter = lambda line: line.startswith('>')
    for is_header, block in groupby(handle, delimiter):
        if is_header:
            header = next(block)[1:].rstrip()
        else:
            yield(header, joiner.join(line.rstrip() for line in block))



def annotate_structures(input_file, output_file):
    """ Annotate secondary structure predictions with structural contexts.

    Given dot-bracket strings this function will annote every character
    as either 'H' (hairpin), 'S' (stem), 'I' (internal loop/bulge) or 'M' (multi loop). The input file
    must be a fasta formatted file and each sequence and structure must span a single line:

    '>header
    'CCCCAUAGGGG
    '((((...)))) (-3.3)

    This is the default format of e.g. RNAfold. The output file will contain the annotated string:

    '>header
    'CCCCAUAGGGG
    'SSSSHHHSSSS

    Parameters
    ----------
    input_file : str
        A fasta file containing secondary structure predictions.
    
    output_file : str
        A fasta file with secondary structure annotations.
    """
    handle_in = get_handle(input_file, "rt")
    handle_out = get_handle(output_file, "wt")
    for header, entry in parse_fasta(handle_in, "_"):
        entry = entry.split("_")
        bg = cgb.BulgeGraph()
        bg.from_dotbracket(entry[1].split()[0])
        handle_out.write(">{}\n".format(header))
        handle_out.write("{}\n{}\n".format(entry[0], bg.to_element_string().upper()))
    handle_in.close()
    handle_out.close()


def predict_structures(input_file, output_file, num_processes=None, annotate=False):
    """ Predict secondary structures for RNA sequences.

    This is a convenience function to get quick RNA secondary structure predictions. The function
    will try to use the ViennaRNA python bindings or the RNAfold binary to perform predictions. If
    neither can be found the function raises a RuntimeError. Using the ViennaRNA python bindings
    is preferred as it is faster.

    Entries of the output file look as follows if annotate = False:

    '>header
    'CCCCAUAGGGG
    '((((...)))) (-3.3)

    If annotate = True the annotated structure string instead of the dot-bracket string will be printed:

    '>header
    'CCCCAUAGGGG
    'SSSSHHHSSSS

    Have a look at the annotate_structures() function for more information about annotated structure strings.

    Warning: Due to the way Python works spinning up additional processes means copying the complete
    memory of the original process, i.e. if the original processes already uses 5 GB of RAM each additional
    process will use 5 GB as well.

    Parameters
    ----------
    input_file : str
        A fasta file with RNA sequences.
    
    output_file : str
        A fasta file with sequences and structures.
    
    num_processes : int
        The number of parallel processes to use for prediction. (default: number of available cores)
    
    annotate : bool
        Output the annotated structure string instead of the dot-bracket string. (default: false)
    """
    from multiprocessing import Pool

    try:
        from RNA import fold
        predictor = _predict_rnalib
    except:
        if which("RNAfold") == None:
            raise RuntimeError("Error: Neither ViennaRNA python bindings nor RNAfold executable found.")
        predictor = _predict_binary
    handle = get_handle(input_file, "rt")
    if num_processes == None:
        num_processes = max(1, int(os.cpu_count()/2))
    with Pool(num_processes) as pool:
        if annotate:
            data = pool.starmap(func = _predict_and_annotate, 
                                iterable = zip(parse_fasta(handle), repeat(predictor)),
                                chunksize = 2)
            formatter = ">{}\n{}\n{}\n"
        else:
            data = list(pool.imap(func = predictor,
                                  iterable = parse_fasta(handle),
                                  chunksize = 2))
            formatter = ">{}\n{}\n{} ({})\n"
    handle.close()
    handle = get_handle(output_file, "wt")
    for entry in data:
        handle.write(formatter.format(*entry))
    handle.close()


def _predict_rnalib(fasta_entry):
    from RNA import fold
    return (*fasta_entry, *fold(fasta_entry[1]))


def _predict_binary(fasta_entry):
    out = check_output("echo {} | RNAfold --noPS".format(fasta_entry[1]), shell = True).split()
    return (*fasta_entry, out[1].decode("utf-8"), float(out[-1][:-1].replace(b"(", b"")))


def _predict_and_annotate(fasta_entry, predict_function):
    predict_entry = predict_function(fasta_entry)
    bg = cgb.BulgeGraph()
    bg.from_dotbracket(predict_entry[2])
    return (predict_entry[0], predict_entry[1], bg.to_element_string().upper())


def auROC(labels, predictions):
    from sklearn.metrics import auc, roc_curve
    fpr, tpr, _ = roc_curve(labels, predictions)
    return fpr, tpr, auc(fpr, tpr)


def auPR(labels, predictions):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    precision, recall, _ = precision_recall_curve(labels, predictions)
    return precision, recall, average_precision_score(labels, predictions)


def performance_report(labels, predictions):
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import precision_recall_fscore_support
    classes =  list(range(labels.shape[1]))
    roc_aucs, pr_aucs  = [], []
    if len(classes) == 2:
        roc_aucs = [auROC(labels[:, 0], predictions[:, 0])[2]] * 2
        pr_aucs = [auPR(labels[:, 0], predictions[:, 0])[2]] * 2
        labels = label_binarize(np.argmax(labels, axis = 1), classes = classes)
    else:
        for x in classes:
            roc_aucs.append(auROC(labels[:, x], predictions[:, x])[2])
            pr_aucs.append(auPR(labels[:, x], predictions[:, x])[2])
    if not np.isclose(np.sum(predictions, axis=1), 1).all():
        # multi-label classification
        y_pred = predictions > 0.5
        y_pred.dtype = np.uint8
    else:
        y_pred = label_binarize(np.argmax(predictions, axis = 1), classes = classes)
    prec_recall_f1_support = precision_recall_fscore_support(labels, y_pred)
    report = np.empty((len(classes), 6))
    for x in classes:
        report[x,:] = [prec_recall_f1_support[0][x], prec_recall_f1_support[1][x],
                       prec_recall_f1_support[2][x], roc_aucs[x],
                       pr_aucs[x], prec_recall_f1_support[3][x]]
    return report


def get_performance_report(labels, predictions):
    """ Get a performance overview of a classifier.

    The report contains precision, recall, f1-score, ROC-AUC and Precision-Recall-AUC for every
    class (in a 1 vs. all approach) and weighted averages (weighted by the the number
    of sequences 'n' in each class).

    Parameters
    ----------
    labels : numpy.ndarray
        A binary matrix of shape (num sequences, num classes) containing the true labels.
    
    predictions : numpy.ndarray
        A matrix of shape (num sequences, num classes) containing predicted probabilites.
    
    Returns
    -------
    report : str
        Summary table of the above mentioned performance measurements.
    """
    classes =  list(range(labels.shape[1]))
    report = performance_report(labels, predictions)
    out = []
    out.append("             precision    recall  f1-score   roc-auc    pr-auc          n")
    formatter = "{:>12}" + "{:>10.3f}" * 5 + "  |" + "{:>8}"
    for x in classes:
        out.append(formatter.format("class_{}".format(x), *report[x, 0:-1], int(report[x,-1])))
    out.append('\n')
    out.append(formatter.format(
        "weighted avg",
        *np.sum(report[:,0:-1] * report[:,-1, np.newaxis], axis=0)/int(sum(report[:,-1])),
        " "
    ))
    out.append('\n')
    return '\n'.join(out)



def plot_roc(labels, predictions, file_path):
    """ Get ROC curves for every class.

    In the case of more than two classes the comparisons will be performed in a 1 vs. all
    approach (i.e. you get one curve per class).

    Parameters
    ----------
    labels : numpy.ndarray
        A binary matrix of shape (num sequences, num classes) containing the true labels.
    
    predictions : numpy.ndarray
        A matrix of shape (num sequences, num classes) containing predicted probabilites.
    
    file_path : str
        The file the plot should be saved to.
    """
    classes = list(range(labels.shape[1]))
    colors = _get_colors(len(classes))
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (4.6666,4))
    _hide_top_right(ax)
    ax.plot([0, 1], [0, 1], color = 'black', linewidth = 1, linestyle = '--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    if len(classes) == 2:
        fpr, tpr, roc_auc = auROC(labels[:, 0], predictions[:, 0])
        label = 'AUC = {:.3f}'.format(roc_auc)
        ax.plot(fpr, tpr, linewidth = 2.2, color = colors[0], label = label)
    else:
        for x in classes:
            fpr, tpr, roc_auc = auROC(labels[:, x], predictions[:, x])
            label = 'AUC class_{} = {:.3f}'.format(x, roc_auc)
            ax.plot(fpr, tpr, linewidth = 2.2, color = colors[x], label = label)
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., framealpha=1)
    fig.savefig(file_path, bbox_inches = 'tight')
    plt.close(fig)


def plot_prec_recall(labels, predictions, file_path):
    """ Get Precision-Recall curves for every class.

    In the case of more than two classes the comparisons will be performed in a 1 vs. rest
    approach (i.e. you get one curve per class).

    Parameters
    ----------
    labels : numpy.ndarray
        A binary matrix of shape (num sequences, num classes) containing the true labels.
    
    predictions : numpy.ndarray
        A matrix of shape (num sequences, num classes) containing predicted probabilites.
    
    file_path : str
        The file the plot should be saved to.
    """
    classes = list(range(labels.shape[1]))
    colors = _get_colors(len(classes))
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (4.6666,4))
    _hide_top_right(ax)
    ax.plot([0, 1], [0, 1], color = 'white', linewidth = 1, linestyle = '--')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    if len(classes) == 2:
        precision, recall, prec_auc = auPR(labels[:,0], predictions[:, 0])
        label = 'AUC = {:.3f}'.format(prec_auc)
        ax.plot(recall, precision, linewidth = 2.2, color = colors[0], label = label)
    else:
        for x in classes:
            precision, recall, prec_auc = auPR(labels[:, x], predictions[:, x])
            label = 'AUC class_{} = {:.3f}'.format(x, prec_auc)
            ax.plot(recall, precision, linewidth = 2.2, color = colors[x], label = label)
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., framealpha=1)
    fig.savefig(file_path, bbox_inches = 'tight')
    plt.close(fig)



def save_as_meme(logos, file_path):
    """ Save sequence (or structure) motifs in MEME format.

    Parameters
    ----------
    logos : [pysster.Motif]
        A list of Motif objects.

    file_path : str
        The name of the output text file.
    """
    alphabet = logos[0].alphabet
    with open(file_path, "wt") as handle:
        handle.write("MEME version 4\n\nALPHABET= {}\n\nstrands: + -\n\n".format(alphabet))
        handle.write("Background letter frequencies (from uniform background):\n")
        header = ""
        for c in alphabet:
            header += "{} {:7.5f} ".format(c, 1/len(alphabet))
        handle.write(header[:-1] + '\n')
        for i, logo in enumerate(logos):
            pwm = logo.pwm
            handle.write("\nMOTIF motif_{} motif_{}\n\n".format(i,i))
            handle.write("letter-probability matrix: alength= {} w= {} nsites= 20 E= 0\n".format(
                len(alphabet),
                pwm.shape[0]
            ))
            for row in range(pwm.shape[0]):
                handle.write("  {}\t\n".format("\t  ".join(str(round(x, 6)) for x in pwm[row,:])))


def run_tomtom(motif_file, output_folder, database, options = None):
    """ Compare a MEME file against a database using TomTom.

    Default options string: "-min-overlap 5 -verbosity 1 -xalph -evalue -thresh 0.1"

    Parameters
    ----------
    motif_file : str
        A MEME file.
    
    output_folder : str
        The folder the TomTom output will be saved in.
    
    database : str
        A MEME file serving as the database to compare against.
    
    option : str
        Command line options passed to TomTom.
    """
    if which("tomtom") == None:
        raise RuntimeError("Error: tomtom executable not found.")
    if output_folder[-1] != "/":
        output_folder += "/"
    if not os.path.isdir(output_folder):  
        os.makedirs(output_folder)
    if options == None:
        options = "-min-overlap 5 -verbosity 1 -xalph -evalue -thresh 0.1"
    command = "tomtom {}".format(options)
    call("{} -oc {} {} {}".format(command, output_folder, motif_file, database), shell = True)


def softmax(x):
    x = np.exp(x - np.max(x))
    return x / x.sum(axis = 0)


def _hide_top_right(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')


def plot_motif_summary(position_max, mean_acts, kernel, file_path):
    from PIL import Image
    classes = []
    ylim_hist, ylim_mean = 0, 0
    for i, hist in enumerate(position_max):
        if len(hist) == 0:
            print("Warning: class {} did not activate kernel {}. No plots were created.".format(
                i, kernel
            ))
        else:
            classes.append(i)
            ylim_hist = max(ylim_hist, Counter(hist).most_common(1)[0][1])
            ylim_mean = max(ylim_mean, max(mean_acts[i][0] + mean_acts[i][1]))
    xlim = len(mean_acts[classes[0]][0]) + 1
    old_fontsize = matplotlib.rcParams['font.size']
    matplotlib.rcParams.update({'font.size': 30})
    files = []
    n_per_plot = 3
    n_plots = ceil(len(classes)/n_per_plot)
    class_idx = -1
    for plot_id in range(n_plots):
        classes_left = len(classes) - plot_id*n_per_plot
        classes_this_plot = min(n_per_plot, classes_left)
        fig, ax = plt.subplots(nrows = 2, 
                               ncols = classes_this_plot,
                               figsize = (19*classes_this_plot, 12))
        for class_num in range(classes_this_plot):
            class_idx += 1
            # histograms
            ax.flat[class_num].set_ylim((0, ylim_hist))
            ax.flat[class_num].hist(position_max[classes[class_idx]], histtype="stepfilled",
                                    bins = xlim, range = (0, xlim))
            ax.flat[class_num].set_xlabel("sequence position")
            ax.flat[class_num].set_ylabel("counts")
            ax.flat[class_num].set_title("kernel {}, class_{}, (n = {})".format(
            kernel, classes[class_idx], len(position_max[classes[class_idx]])
            ))
            _hide_top_right(ax.flat[class_num])
            # mean activations
            ax.flat[class_num + classes_this_plot].set_ylim((0, ylim_mean))
            ax.flat[class_num + classes_this_plot].fill_between(list(range(1, xlim)),
                                                                mean_acts[classes[class_idx]][0] - mean_acts[classes[class_idx]][1],
                                                                mean_acts[classes[class_idx]][0] + mean_acts[classes[class_idx]][1],
                                                                alpha = 0.1)
            ax.flat[class_num + classes_this_plot].plot(list(range(1, xlim)),
                                                        mean_acts[classes[class_idx]][0],
                                                        linewidth = 5.0)
            ax.flat[class_num + classes_this_plot].set_xlabel("sequence position")
            ax.flat[class_num + classes_this_plot].set_ylabel("activation")
            _hide_top_right(ax.flat[class_num + classes_this_plot])
        fig.tight_layout()
        files.append("{}/plotsum{}.png".format(gettempdir(), plot_id))
        fig.savefig(files[-1])
        plt.close(fig) # fig.clf() before close() seems to release memory faster
    matplotlib.rcParams.update({'font.size': old_fontsize})
    if len(files) == 1:
        move(files[0], file_path)
    else:
        images = []
        for file_name in files:
            images.append(Image.open(file_name))
        combine_images(images, file_path)
        for file_name in files:
            remove(file_name)


def plot_violins(data, kernel, file_path):
    old_fontsize = matplotlib.rcParams['font.size']
    matplotlib.rcParams.update({'font.size': 15})
    num_plots = len(data)
    labels = ["class_{}".format(x) for x in range(num_plots)]
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (max(5, num_plots), 5))
    _hide_top_right(ax)
    ax.grid(axis = "y", alpha = 0.3)
    ax.set_title("Activations, kernel {}".format(kernel))
    ax.set_ylabel("max activations")
    parts = ax.violinplot(data, showmeans = True, showextrema = True)
    ax.set_ylim(bottom = 0) 
    parts['cmeans']._linewidths = [2]
    parts['cmins']._linewidths = [2]
    parts['cmaxes']._linewidths = [2]
    parts['cbars']._linewidths = [2]
    ax.get_xaxis().set_tick_params(length=0)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    plt.xticks(rotation=90)
    fig.subplots_adjust(bottom=0.2)
    plt.tight_layout()
    fig.savefig(file_path)
    plt.close(fig)
    matplotlib.rcParams.update({'font.size': old_fontsize})


def plot_motif(logo, file_path, colors_sequence, colors_structure):
    from PIL import Image
    if isinstance(logo, tuple):
        img1 = logo[0].plot(colors_sequence, scale=0.75)
        img2 = logo[1].plot(colors_structure, scale=0.75)
        img = Image.new("RGB", (img1.size[0], img1.size[1]+img2.size[1]))
        img.paste(img1, (0, 0))
        img.paste(img2, (0, img1.size[1]))
        img1.close()
        img2.close()
    else:
        img = logo.plot(colors_sequence, scale=0.75)
    img.save(file_path)
    img.close()
    plt.close('all')


def _set_sns_context(n_kernel):
    import seaborn as sns
    if n_kernel <= 25:
        sns.set_context("notebook", rc={"ytick.labelsize":26})
    elif 25 < n_kernel <= 50:
        sns.set_context("notebook", rc={"ytick.labelsize":22})
    elif 50 < n_kernel <= 75:
        sns.set_context("notebook", rc={"ytick.labelsize":14})
    elif 75 < n_kernel <= 100:
        sns.set_context("notebook", rc={"ytick.labelsize":8})
    else:
        sns.set_context("notebook", rc={"ytick.labelsize":5})


def _get_colors(x):
    import seaborn as sns
    palette = ["hls", "Set1"][x < 10]
    return sns.color_palette(palette, x, 0.6)


def _plot_heatmap(file_path, data, class_id, classes = None):
    import seaborn as sns
    _set_sns_context(data.shape[1])
    n_classes = len(set(class_id))
    palette = _get_colors(n_classes)
    colors = [palette[x] for x in class_id]
    g = sns.clustermap(data = data.T, method = "ward", metric = "euclidean",
                       cmap = "RdBu_r", xticklabels = False, yticklabels = True,
                       figsize = (30,25), row_cluster = True, col_cluster = True,
                       linewidths = 0, col_colors = colors, robust = True,
                       z_score = 0, cbar_kws={"ticks":[-1.5,0,+1.5]})
    g.ax_col_dendrogram.set_xlim([0,1e-10])
    g.ax_col_dendrogram.set_ylim([0,1e-10])
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
    sns.set(font_scale=2.8)
    if classes == None:
        classes = list(range(n_classes))
    for x in range(n_classes):
        g.ax_col_dendrogram.bar(0, 0, color=palette[x],
                                label="class_{}".format(classes[x]), linewidth=0)
    g.ax_col_dendrogram.legend(loc = "center", ncol = min(6, n_classes))
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=25)
    plt.savefig(file_path, bbox_inches = 'tight')
    plt.close('all')
    sns.reset_orig()


def combine_images(images, output_file):
    from PIL import Image
    widths, heights = zip(*(i.size for i in images))
    new_im = Image.new('RGB', (max(widths), sum(heights)), "#ffffff")
    y_offset = 0
    for im in images:
        new_im.paste(im, (0, y_offset))
        im.close()
        y_offset += im.size[1]
    new_im.save(output_file)
    new_im.close()
    plt.close('all')


# helper function; np.argmax always returns the first occurrence of the max value
# even if it occurs multiple times; this function randomly selects one of those instead
def randargmax(data):
    rtol, atol = 1e-09, 0.0
    result = np.empty((data.shape[0],), dtype=np.uint32)
    max_val = np.max(data, axis=1)
    for x in range(data.shape[0]):
        result[x] = np.random.choice(
            np.where(abs(data[x, ] - max_val[x]) <= np.maximum(rtol * np.maximum(abs(data[x, ]), abs(max_val[x])), atol))[0])
    return result


def html_report(sorted_idx, scores, folder, class_num, size=None):
    handle = open(folder+"summary.html", "wt")
    handle.write('<!doctype html>\n<html>\n<head>\n<meta charset="UTF-8">\n')
    handle.write('<title>Kernel Summary</title>\n<style media="screen" type="text/css">\n')
    handle.write('#report {white-space: nowrap;}\n')
    handle.write('td {text-align: center; font-weight: bold; padding: 20px;}\n')
    handle.write('table {margin: 0 auto; border-collapse: collapse;}\n')
    handle.write('tr:nth-child(even) {background-color: #f2f9ff;}</style>\n</head>\n')
    handle.write('<body>\n<div id="report">\n<table>\n')
    for kernel in sorted_idx:
        handle.write('<tr>\n<td>Kernel {}<br/>score = {:.3f}</td>\n'.format(kernel, scores[kernel]))
        if size == None:
            handle.write('<td><img src="motif_kernel_{}.png" height=150/></td>\n'.format(kernel))
        else:
            handle.write('<td><img src="motif_kernel_{}.png" height=150/><br>'.format(kernel))
            handle.write('<img src="additional_features_kernel_{}.png" height={}/></td>\n'.format(kernel, size))
        handle.write('<td><img src="activations_kernel_{}.png" height=300/></td>\n'.format(kernel))
        handle.write('<td><img src="position_kernel_{}.png" height={}/></td>\n</tr>\n'.format(
            kernel, 300 * max(ceil(class_num/3), 1)))
    handle.write('</table>\n</div>\n</body>\n</html>')
    handle.close()


def plot_positionwise(add_data, identifiers, file_path):
    old_fontsize = matplotlib.rcParams['font.size']
    matplotlib.rcParams.update({"font.size": 30})
    x = list(range(1, add_data[0][0].shape[0]+1))
    fig, ax = plt.subplots(nrows=len(add_data), ncols=1,
                           figsize=(int(26*(len(x)/20)), 7*len(add_data)))
    if not isinstance(ax, np.ndarray):
        ax = [ax]
    for i in range(len(add_data)):
        mean, std = np.mean(add_data[i], axis=0), np.std(add_data[i], axis=0)
        ax[i].fill_between(x, mean-std, mean+std, color='orange', alpha = 0.1)
        ax[i].plot(x, mean, 'o-', linewidth=5.0, markersize=15.0, color="orange")
        if len(x) <= 30:
            ax[i].set_xticks(x)
        elif len(x) > 75:
            ax[i].set_xticks(list(range(1, max(x), 3)))
        else:
            ax[i].set_xticks(list(range(1, max(x), 2)))
        ax[i].set_ylabel("{}".format(identifiers[i]))
        _hide_top_right(ax[i])
    plt.tight_layout()
    fig.savefig(file_path)
    plt.close(fig)
    matplotlib.rcParams.update({"font.size": old_fontsize})