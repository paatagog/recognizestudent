import math
from sklearn import neighbors, svm
import os
import os.path
import pickle
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
from datetime import date

TRAIN_DIR = '../Data/train'
MODEL_SAVE_PATH_KNN = 'models/knn/'
MODEL_SAVE_PATH_SVC = 'models/svc/'


def get_encodings():
    encodings = []
    persons = []

    # Loop through each person in the training set
    for class_dir in os.listdir(TRAIN_DIR):
        if not os.path.isdir(os.path.join(TRAIN_DIR, class_dir)):
            continue

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(TRAIN_DIR, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            # Checking if image contains one face
            if len(face_bounding_boxes) == 1:
                encodings.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                persons.append(class_dir)
                # break
    return encodings, persons


def save_clf(clf, clf_name):
    day = date.today()

    if clf_name == 'knn':
        path = MODEL_SAVE_PATH_KNN + day.strftime("%m-%d-%y") + '-model.clf'
    else:
        path = MODEL_SAVE_PATH_SVC + day.strftime("%m-%d-%y") + '-model.clf'

    if path is not None:
        with open(path, 'wb') as f:
            pickle.dump(clf, f)


def train_knn(knn_algo='ball_tree'):
    encodings, persons = get_encodings()

    # Determine how many neighbors to use for weighting in the KNN classifier
    n_neighbors = int(round(math.sqrt(len(encodings))))

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(encodings, persons)

    save_clf(knn_clf, 'knn')

    return knn_clf


def train_svc():
    encodings, persons = get_encodings()

    # Create and train the SVC classifier
    svc_clf = svm.SVC(gamma='scale')
    svc_clf.fit(encodings, persons)

    save_clf(svc_clf, 'svc')

    return svc_clf


(enc, cls) = get_encodings()
train_knn()
# print(enc[0], cls[0])
