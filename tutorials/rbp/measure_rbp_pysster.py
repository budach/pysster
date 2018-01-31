import os
from pysster.Data import Data
from pysster.Model import Model
from pysster import utils
from time import time


def main():

    RBPs = [("data/pum2.train.positive.fasta",
             "data/pum2.train.negative.fasta",
             "data/pum2.test.positive.fasta",
             "data/pum2.test.negative.fasta",
             "PUM2"),
            ("data/qki.train.positive.fasta",
             "data/qki.train.negative.fasta",
             "data/qki.test.positive.fasta",
             "data/qki.test.negative.fasta",
             "QKI"),
            ("data/igf2bp123.train.positive.fasta",
             "data/igf2bp123.train.negative.fasta",
             "data/igf2bp123.test.positive.fasta",
             "data/igf2bp123.test.negative.fasta",
             "IGF2BP123"),
            ("data/srsf1.train.positive.fasta",
             "data/srsf1.train.negative.fasta",
             "data/srsf1.test.positive.fasta",
             "data/srsf1.test.negative.fasta",
             "SRSF1"),
            ("data/taf2n.train.positive.fasta",
             "data/taf2n.train.negative.fasta",
             "data/taf2n.test.positive.fasta",
             "data/taf2n.test.negative.fasta",
             "TAF2N"),
            ("data/nova.train.positive.fasta",
             "data/nova.train.negative.fasta",
             "data/nova.test.positive.fasta",
             "data/nova.test.negative.fasta",
             "NOVA")]

    for entry in RBPs:
        output_folder = entry[4] + "_pysster/"
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

        start = time()

        # predict secondary structures
        utils.predict_structures(entry[0], entry[0]+".struct.gz", annotate=True)
        utils.predict_structures(entry[1], entry[1]+".struct.gz", annotate=True)
        utils.predict_structures(entry[2], entry[2]+".struct.gz", annotate=True)
        utils.predict_structures(entry[3], entry[3]+".struct.gz", annotate=True)

        # load data
        data = Data([entry[0]+".struct.gz", entry[1]+".struct.gz"], ("ACGU", "HIMS"))
        data.train_val_test_split(0.8, 0.1999) # we need to have at least one test sequence, even though we don't need it
        print(data.get_summary())

        # training
        params = {"kernel_len": 8}
        model = Model(params, data)
        model.train(data)

        # load and predict test data
        data_test = Data([entry[2]+".struct.gz", entry[3]+".struct.gz"], ("ACGU", "HIMS"))
        predictions = model.predict(data_test, "all")

        stop = time()
        print("{}, time in seconds: {}".format(entry[4], stop-start))

        # performance evaluation
        labels = data_test.get_labels("all")
        utils.plot_roc(labels, predictions, output_folder+"roc.pdf")
        utils.plot_prec_recall(labels, predictions, output_folder+"prec.pdf")
        print(utils.get_performance_report(labels, predictions))

        # get motifs
        activations = model.get_max_activations(data_test, "all")
        logos, scores = [], []
        for kernel in range(model.params["kernel_num"]):
            logo, score = model.visualize_kernel(activations, data_test, kernel, output_folder)
            logos.append(logo)
            scores.append(score)
        
        # sort motifs by importance score
        sorted_idx = [i[0] for i in sorted(enumerate(scores), key=lambda x:x[1])]
        with open(output_folder+"kernel_scores.txt", "wt") as handle:
            for x in sorted_idx:
                print("kernel {:>3}: {:.3f}".format(x, scores[x]))
                handle.write("kernel {:>3}: {:.3f}\n".format(x, scores[x]))

        # save model to drive
        utils.save_model(model, "{}model.pkl".format(output_folder))


if __name__ == "__main__":
    main()
