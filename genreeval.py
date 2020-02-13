import collections
from IPython.display import HTML
import pandas
import sys
import json
import csv
import os
from io import StringIO
import datetime


class Result(object):

    def __init__(self, data, confusion_matrix, raw_accuracy, stats):
        self.confusion_matrix = confusion_matrix
        self.raw_accuracy = round(raw_accuracy, 2)
        self.data = data
        self.stats = stats
        self.date = str(datetime.datetime.now().replace(microsecond=0))

    def save(self, directory=None):
        confusion = StringIO()
        self.confusion_matrix.to_csv(confusion)

        data = {"raw_accuracy": self.raw_accuracy,
                "confusion_matrix": confusion.getvalue(),
                "data": self.data,
                "stats": self.stats,
                "date": self.date}
        d = self.data
        fname = "%s-%s-%s-%s.json" % (d["name"], d["strategy"], d["tags"], d["duplicates"])
        if directory:
            fname = os.path.join(directory, fname)
        fp = open(fname, "w")
        json.dump(data, fp)

    @property
    def confusion_matrix_percent(self):
        return confusion_matrix_percent(self.confusion_matrix)

    @property
    def normalised_accuracy(self):
        return normalised_accuracy(self.confusion_matrix)

    @property
    def html(self):
        d = self.data
        name = "%s %s-%s-%s" % (d["name"], d["strategy"], d["tags"], d["duplicates"])
        return confusion_matrix_html(
                confusion_matrix_percent(self.confusion_matrix),
                name)

    @property
    def latex(self):
        return confusion_matrix_latex(
                confusion_matrix_percent(self.confusion_matrix),
                self.data, dict(self.stats["classes_groundtruth"]))

def load_result(filename):
    j = json.load(open(filename))

    confusion = StringIO(j["confusion_matrix"])
    confusion_matrix = pandas.read_csv(confusion, index_col=0)

    result = Result(j["data"], confusion_matrix, j["raw_accuracy"],
            j["stats"])
    result.date = j["date"]
    return result


class Evaluator(object):
    def __init__(self, groundtruth_file, gt_map, estimates_file, name, description):
        """ Args:
            ground_truth: dict: {mbid: [(tag, weight), (tag, weight), ...]}
            gt_map: dict: {estimate classname: ['possible', 'gt_classes']}
            estimates: filename: csv, contains at least 'mbid' and
              'estimate classname' fields.
        """
        self.groundtruth_file = groundtruth_file
        self.gt_map = gt_map
        self.estimates_file = estimates_file
        self.name = name
        self.description = description

    def load(self):
        print("Loading estimates...")
        # Load AcousticBrainz estimate data from csv
        estimate_classes = self.gt_map.keys()
        estimate_df = pandas.read_json(self.estimates_file, orient='index')

        # estimate_df = estimate_df.set_index("mbid")
        print("done")

        print("Loading ground truth...")
        gt = json.load(open(self.groundtruth_file))
        self.gt_classes = set([item for sublist in self.gt_map.values() for item in sublist])

        # Remove speech and other
        #self.ground_truth = prefilter_gt(gt)
        self.ground_truth = gt
        self.estimates = estimate_df[estimate_df.index.isin(self.ground_truth.keys())]

        print("done")

    def evaluate(self, strategy, tags, duplicates):
        """ args: strategy:   S1,S2
                  tags:       FIRST,ALL,ONLY
                  duplicates: D1,D2
        """
        if strategy == "S1":
            ground_truth = filter_gt(self.ground_truth, self.gt_classes)
        elif strategy == "S2":
            ground_truth = self.ground_truth
        else:
            raise Exception("Unknown strategy")

        if tags == "ALL":
            # This is the same as not filtering it
            pass
        elif tags == "ONLY":
            ground_truth = filter_gt_onetag(ground_truth)
        elif tags == "FIRST":
            ground_truth = filter_gt_firsttag(ground_truth)
        else:
            raise Exception("Unknown tag strategy")

        if duplicates == "D2":
            estimates = self.estimates
        elif duplicates == "D1":
            estimates = deduplicate_estimates(self.estimates)
        else:
            raise Exception("Unknown duplicate strategy")

        # If we chose S1 we should re-filter estimates to remove
        # those which are not in our GT
        estimates = estimates[estimates.index.isin(ground_truth.keys())]

        print("%s - %s - %s - %s" % (self.name, strategy, tags, duplicates))

        confusion_matrix, raw_accuracy = create_confusion_matrix(estimates, ground_truth, self.gt_map)

        stats = self.get_stats(ground_truth)
        stats["len_estimates"] = len(estimates)
        data = {"name": self.name,
                "description": self.description,
                "strategy": strategy,
                "tags": tags,
                "duplicates": duplicates,
                "gtmap": self.gt_map,
                "estimates_file": self.estimates_file,
                "groundtruth_file": self.groundtruth_file}
        self.result = Result(data, confusion_matrix, raw_accuracy, stats)
        print("")
        return self.result

    def get_stats(self, ground_truth):
        class_counts = collections.Counter()
        for k, v in ground_truth.items():
            for t, w in v:
                class_counts[t] += 1
        lengt = len(ground_truth.keys())
        #print "Number of tracks in ground truth:", lengt
        #print "Tag counts in ground truth:"
        #for cls, cnt in class_counts.most_common():
        #    print "  %s: %s" % (cls, cnt)
        return {"len_groundtruth": lengt,
                "classes_groundtruth": class_counts.most_common()}

def run_evaluations(ground_truth, gt_map, estimates, name, description):
    """
    Args:
      ground_truth: filename: {mbid: [(tag, weight), (tag, weight), ...]}
      gt_map: {estimate classname: ['possible', 'gt_classes']}
      estimates: filename: csv, contains at least 'mbid' and
                 'estimate classname' fields.
    """

    evals = {}
    evals[name] = {}

    for strategy in ["S1", "S2"]:
        evals[name][strategy] = {}
        for tags in ["ALL", "ONLY"]: #, "FIRST"]:
            evals[name][strategy][tags] = {}
            for d in ["D1", "D2"]:
                print("RUNNING EVALUATION: %s %s %s %s" % (name, strategy, tags, d))
                evaluator = Evaluator(ground_truth, gt_map, estimates, name, description)
                evaluator.load()
                evaluator.evaluate(strategy, tags, d)
                evals[name][strategy][tags][d] = evaluator.result
    return evals


def prefilter_gt(ground_truth):
    # This is a list of all of our ground-truth classes except for
    # 'comedy' and 'other'
    tokeep = ['african', 'asian', 'avant-garde', 'blues',
            'caribbean and latin american', 'classical', 'country',
            'easy listening', 'electronic', 'folk', 'hip hop',
            'jazz', 'pop', 'rhythm and blues', 'rock', 'ska',
            'dance', 'reggae', 'disco', 'alternative rock',
            'metal',
            'cla', 'dan', 'hip', 'jaz', 'pop', 'rhy', 'roc',  # Rosamerica classees without cleaning
            'alternative', 'blues', 'electronic', 'folkcountry', 'funksoulrnb', 'jazz', 'pop', 'raphiphop', 'rock',
            'blu', 'cla', 'cou', 'dis', 'hip', 'jaz', 'met', 'pop', 'reg', 'roc']  # Tza classees without cleaning
    return filter_gt(ground_truth, tokeep)

def filter_gt(ground_truth, tokeep):
    """ Remove all classes in ground_truth which are not in `tokeep` """
    gt = {}
    for k, v in ground_truth.items():
        tags = []
        for t, w in v:
            if t in tokeep:
                tags.append([t, w])
        if tags:
            gt[k] = tags

    return gt

def deduplicate_estimates(testset):
    counts = collections.Counter()
    for k in testset.index:
        counts[k] += 1
    single_mbids = set()
    for k, v in counts.most_common():
        if v == 1:
            single_mbids.add(k)
    #print "*******"
    #print name
    #print "number of recordings", len(testset.index)
    #print "number of recordings with no dups", len(single_mbids)
    testset_nodup = testset[testset.index.isin(single_mbids)]
    #print "Percentage of recordings with no duplicates:", len(testset_nodup.index)*100.0/len(testset.index)
    return testset_nodup

def filter_gt_onetag(ground_truth):
    """ Remove all ground truth items that have more than 1 tag """
    gt = {}
    for k, v in ground_truth.items():
        if len(v) == 1:
            gt[k] = v

    return gt

def filter_gt_firsttag(ground_truth):
    """ Select the first tag for each ground truth item """
    gt = {}
    for k, v in ground_truth.items():
        gt[k] = [v[0]]

    return gt

def create_confusion_matrix(estimates, ground_truth, gt_map):
    """
    Args:
       estimates: estimate data from acousticbrainz
       ground_truth: the ground truth
       gt_map: a dict of {estimate_classname: ground truth classname}
    """
    confusion = collections.defaultdict(collections.Counter)
    correct = 0
    c = 0
    ct = 0
    pbar = ProgressBar(len(estimates))
    for i in estimates.iterrows():
        estimate = i[1][0]
        mbid = i[0]

        # workarround 
        if type(mbid) is int:
            mbid = str(mbid)

        actuals = [a[0] for a in ground_truth[mbid]]
        if estimate in gt_map:
            gt_estimates = gt_map[estimate]
            if not isinstance(gt_estimates, list):
                gt_estimates = [gt_estimates]

            if set(gt_estimates) & set(actuals):
                correct += 1

            for e in gt_estimates:
                for a in actuals:
                    confusion[a][e] += 1

            c += 1
        ct += 1
        if ct % 10000 == 0:
            pbar.animate(ct)
    pbar.animate(ct)
    accuracy = 0
    if c:
        accuracy = correct*100.0/c
        #print "raw accuracy:", accuracy
    else:
        print("error? No items counted")
    confusion_dict = {}
    for k, v in confusion.items():
        confusion_dict[k] = pandas.Series(dict(v))
    df = pandas.DataFrame(confusion_dict).transpose()
    df = df.fillna(0)
    return df, accuracy

def normalised_accuracy(matrix):
    copy = matrix.copy(deep=True)
    fields = copy.keys()

    copy.insert(len(copy.keys()), 'total', copy.sum(axis=1))
    copy.insert(len(copy.keys()), 'coeff', copy['total']*1.0/copy['total'].max())
    copy['total'] = copy['total']/copy['coeff']

    norm_total = copy['total'].sum()
    for f in fields:
        copy[f] = copy[f]/copy['coeff']

    correct = 0
    for f in fields:
        num = copy[f][f]
        correct += num
    return round(correct * 100.0 / norm_total, 2)

def confusion_matrix_percent(matrix):
    copy = matrix.copy(deep=1)
    fields = copy.keys()
    if "total" not in copy:
        copy.insert(len(copy.keys()), 'total', copy.sum(axis=1))

    for f in fields:
        copy[f] = copy[f]*100.0/copy['total']
    del copy['total']
    return copy

def confusion_matrix_html(matrix, name=""):
    matrix = matrix.copy(deep=True)
    colnames = list(matrix.keys())
    rownames = list(matrix.index)
    newrows = colnames + [r for r in rownames if r not in colnames]
    matrix = matrix.reindex(newrows)
    fields = matrix.keys()
    head = "<tr><td></td>" + "".join(["<th>%s</th>" % f for f in fields]) + "</tr>"
    data = []
    fields = matrix.keys()
    for gt, row in matrix.iterrows():
        line = "<tr><th>%s</th>" % gt
        for es, pct in row.items():
            pct = round(pct,2)
            bg =""
            if gt == es:
                bg = ' style="background-color:green"'
            elif pct > 10.0:
                bg = ' style="background-color:red"'
            line += "<td%s>%s</td>" % (bg, pct)
        line += "</tr>"
        data.append(line)
    table = "<table>" + head + "\n".join(data) + "</table>"
    if name:
        table = "<b>Confusion matrix for %s</b>" % name + table
    return table
    #return HTML(table)

def confusion_matrix_latex(matrix, meta, classcounts=None):
    matrix = matrix.copy(deep=True)
    colnames = list(matrix.keys())
    rownames = list(matrix.index)
    newrows = colnames + [r for r in rownames if r not in colnames]
    matrix = matrix.reindex(newrows)
    fields = matrix.keys()

    total_tracks = sum(classcounts.values())

    genre_remap = {"caribbean and latin american": "carribean \& latin",
            "rhythm and blues": "rhythm \& blues"}

    numfields = len(fields)
    align = r"l@{\hskip -1em}r|" + "r"*numfields
    header = r"\begin{tabular}{%s}" % align
    legend = r"\multicolumn{2}{c}{Ground-truth} & \multicolumn{%s}{c}{Estimated genre}" % numfields
    renamedfields = [genre_remap.get(f, f) for f in fields]
    fieldnames = "genre & size (\%) & " + " & ".join(renamedfields)
    header = r"""%s
\toprule
%s \\
%s \\
\midrule
    """ % (header, legend, fieldnames)
    data = []
    fields = matrix.keys()
    for gt, row in matrix.iterrows():
        line = "%s & " % genre_remap.get(gt, gt)
        numbers = []
        if gt in classcounts:
            datasetpercent = "%.1f" % round(classcounts[gt] * 100.0/total_tracks, 2)
        else:
            datasetpercent = "xx"
        numbers.append(datasetpercent)
        for es, pct in row.items():
            pct = round(pct, 2)
            if gt == es:
                number = r"\textbf{%.2f}" % pct
            elif pct > 10.0 and pct <= 20:
                number = r"\cellcolor{light-gray}%.2f" % pct
            elif pct > 20.0:
                number = r"\cellcolor{med-gray}%.2f" % pct
            else:
                number = "%.2f" % pct
            numbers.append(number)
        line += " & ".join(numbers)
        line += r" \\"
        data.append(line)

        d = meta
        name = "%s %s-%s-%s" % (d["name"], d["strategy"], d["tags"], d["duplicates"])
    footer = r"""
\bottomrule
\end{tabular}
"""

    table = header + "\n".join(data) + footer
    return table

def table_all_results(name, resultsdir):

    align = "lrrr"
    header = r"\begin{tabular}{%s}" % align
    header = r"""%s
\toprule
 Model & Strategy & Recordings & Accuracy & Normalized \\
 & &  &  & accuracy \\
\midrule
""" % (header, )
    rows = []
    for strat in ["S1", "S2"]:
        for tag in ["ALL", "ONLY"]:
            for dup in ["D1", "D2"]:
                fname = "%s-%s-%s-%s.json" % (name, strat, tag, dup)
                r = load_result(os.path.join(resultsdir, fname))
                raw = round(r.raw_accuracy, 2)
                nor = round(r.normalised_accuracy, 2)
                num = r.stats['len_estimates']
                sname = "%s-%s-%s" % (strat, tag, dup)
                bs = be = ""
                if strat == "S2" and tag == "ONLY" and dup == "D1":
                    bs = r"\textbf{"
                    be = "}"
                data = r"& %s%s%s & %s%s%s & %s%.2f%s & %s%.2f%s \\" % (bs, sname, be, bs, num, be, bs, raw, be, bs, nor, be)
                rows.append(data)
    footer = r"""
\bottomrule
\end{tabular}"""
    table = "%s%s%s" % (header, "\n".join(rows), footer)
    return table

def all_tables():
    dort = table_all_results("Dortmund", "results/dortmund/")
    rosa = table_all_results("Rosamerica", "results/rosamerica")
    rosa30 = table_all_results("Rosamerica30", "results/rosamerica30")
    tzan = table_all_results("Tzanetakis", "results/tzanetakis/")
    lfm = table_all_results("Last.fm", "results/lastfm/")

    open("results/tables/rosamerica-overview.tex", "w").write(rosa)
    open("results/tables/rosamerica30-overview.tex", "w").write(rosa30)
    open("results/tables/tzanetakis-overview.tex", "w").write(tzan)
    open("results/tables/lastfm-overview.tex", "w").write(lfm)
    open("results/tables/dortmund-overview.tex", "w").write(dort)

def confusion_latex():
    # Tables of method/S2-one-D2
    dort = load_result("results/dortmund/Dortmund-S2-ONLY-D1.json")
    rosa = load_result("results/rosamerica/Rosamerica-S2-ONLY-D1.json")
    rosa30 = load_result("results/rosamerica30/Rosamerica30-S2-ONLY-D1.json")
    tzan = load_result("results/tzanetakis/Tzanetakis-S2-ONLY-D1.json")
    lfm = load_result("results/lastfm/Last.fm-S2-ONLY-D1.json")


    open("results/tables/rosamerica-confusion.tex", "w").write(rosa.latex)
    open("results/tables/rosamerica30-confusion.tex", "w").write(rosa30.latex)
    open("results/tables/tzanetakis-confusion.tex", "w").write(tzan.latex)
    open("results/tables/lastfm-confusion.tex", "w").write(lfm.latex)
    open("results/tables/dortmund-confusion.tex", "w").write(dort.latex)

def confusion_html():
    # Tables of method/S2-ONLY-D2
    dort = load_result("results/dortmund/Dortmund-S2-ONLY-D1.json")
    rosa = load_result("results/rosamerica/Rosamerica-S2-ONLY-D1.json")
    rosa30 = load_result("results/rosamerica30/Rosamerica30-S2-ONLY-D1.json")
    tzan = load_result("results/tzanetakis/Tzanetakis-S2-ONLY-D1.json")
    lfm = load_result("results/lastfm/Last.fm-S2-ONLY-D1.json")


    open("results/html/rosamerica-confusion.html", "w").write(rosa.html)
    open("results/html/rosamerica30-confusion.html", "w").write(rosa30.html)
    open("results/html/tzanetakis-confusion.html", "w").write(tzan.html)
    open("results/html/lastfm-confusion.html", "w").write(lfm.html)
    open("results/html/dortmund-confusion.html", "w").write(dort.html)

###################

try:
    from IPython.core.display import clear_output
    have_ipython = True
except ImportError:
    have_ipython = False

class ProgressBar:
    def __init__(self, iterations):
        self.iterations = iterations
        self.prog_bar = '[]'
        self.fill_char = '*'
        self.width = 40
        self.__update_amount(0)
        if have_ipython:
            self.animate = self.animate_ipython
        else:
            self.animate = self.animate_noipython

    def animate_ipython(self, iter):
        print('\r', self,)
        sys.stdout.flush()
        self.update_iteration(iter + 1)

    def animate_noipython( self, iter ):
        print(self, chr( 27 ) + '[A')
        self.update_iteration( iter )

    def update_iteration(self, elapsed_iter):
        self.__update_amount((int(elapsed_iter / float(self.iterations + 1e-6))) * 100.0)
        self.prog_bar += '  %d of %s complete' % (elapsed_iter, self.iterations)

    def __update_amount(self, new_amount):
        percent_done = int(round((new_amount / 100.0) * 100.0))
        all_full = self.width - 2
        num_hashes = int(round((percent_done / 100.0) * all_full))
        self.prog_bar = '[' + self.fill_char * num_hashes + ' ' * (all_full - num_hashes) + ']'
        pct_place = (len(self.prog_bar) // 2) - len(str(percent_done))
        pct_string = '%d%%' % percent_done
        self.prog_bar = self.prog_bar[0:pct_place] + \
            (pct_string + self.prog_bar[pct_place + len(pct_string):])

    def __str__(self):
        return str(self.prog_bar)

