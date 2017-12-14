import sys
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats
from feature_engineering import word_overlap_features
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission

from utils.system import parse_params, check_version

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



def generate_features(stances,dataset,name):
    h, b, y = [],[],[]

    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])

    X_overlap = gen_or_load_feats(word_overlap_features, h, b, "features/overlap."+name+".npy")
    X_refuting = gen_or_load_feats(refuting_features, h, b, "features/refuting."+name+".npy")
    X_polarity = gen_or_load_feats(polarity_features, h, b, "features/polarity."+name+".npy")
    X_hand = gen_or_load_feats(hand_features, h, b, "features/hand."+name+".npy")

    X = np.c_[X_hand, X_polarity, X_refuting, X_overlap]
    return X,y

if __name__ == "__main__":
    #check_version()
    parse_params()

    #Load the training dataset and generate folds
    d = DataSet()
    folds,hold_out = kfold_split(d,n_folds=10)
    
    fold_stances, hold_out_stances = get_stances_for_folds(d,folds,hold_out)
    #print(len(hold_out_stances))
    # Load the competition dataset
    competition_dataset = DataSet("competition_test")
    X_competition, y_competition = generate_features(competition_dataset.stances, competition_dataset, "competition")

    Xs = dict()
    ys = dict()

    # Load/Precompute all features now
    X_holdout,y_holdout = generate_features(hold_out_stances,d,"holdout")
    for fold in fold_stances:
        print(len(fold_stances[fold]))
        Xs[fold],ys[fold] = generate_features(fold_stances[fold],d,str(fold))
        print(len(Xs[fold]))
        print("------------------")

    best_score = 0
    best_fold = None


    classifiers = [
        #DecisionTreeClassifier(max_depth=5),
        #RandomForestClassifier(max_depth=5, n_estimators=100, max_features=1),
        #MLPClassifier(),
        #KNeighborsClassifier(3),
        #AdaBoostClassifier(),
        #GaussianNB(),
        #SVC(kernel="linear", C=0.025),
        #SVC(gamma=2, C=1),
        GradientBoostingClassifier(n_estimators=200, random_state=14128, verbose=True)
        ]


        # 79.15, 76.04, 76.23, 74.52, 78.18, 68.91
    # Classifier for each fold

    for clf in classifiers:
        print(clf)
        for fold in fold_stances:
            ids = list(range(len(folds)))
            del ids[fold]


            for i in ids:
                print(len(Xs[i]))

            X_train = np.vstack(tuple([Xs[i] for i in ids]))
            y_train = np.hstack(tuple([ys[i] for i in ids]))

            X_test = Xs[fold]
            y_test = ys[fold]

            print(X_train.shape)
            #print(X_train[0])

            clf.fit(X_train, y_train)

            predicted = [LABELS[int(a)] for a in clf.predict(X_test)]
            actual = [LABELS[int(a)] for a in y_test]

            fold_score, _ = score_submission(actual, predicted)
            max_fold_score, _ = score_submission(actual, actual)

            score = fold_score/max_fold_score

            print("Score for fold "+ str(fold) + " was - " + str(score))
            if score > best_score:
                best_score = score
                best_fold = clf

            break

        #Run on Holdout set and report the final score on the holdout set
        predicted = [LABELS[int(a)] for a in clf.predict(X_holdout)]
        actual = [LABELS[int(a)] for a in y_holdout]

        print("Scores on the dev set")
        report_score(actual,predicted)
        print("")
        print("")

        #Run on competition dataset
        predicted = [LABELS[int(a)] for a in clf.predict(X_competition)]
        actual = [LABELS[int(a)] for a in y_competition]

        print("Scores on the test set")
        report_score(actual,predicted)



    #Run on Holdout set and report the final score on the holdout set
    predicted = [LABELS[int(a)] for a in best_fold.predict(X_holdout)]
    actual = [LABELS[int(a)] for a in y_holdout]

    print("Scores on the dev set")
    report_score(actual,predicted)
    print("")
    print("")

    #Run on competition dataset
    predicted = [LABELS[int(a)] for a in best_fold.predict(X_competition)]
    actual = [LABELS[int(a)] for a in y_competition]

    print("Scores on the test set")
    report_score(actual,predicted)
