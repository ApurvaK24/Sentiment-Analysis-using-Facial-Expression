# common dependencies
import os
from os import path
import warnings
import time
import pickle
import logging


import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import tensorflow as tf

import realtime
import VGGFace
import  Age, Gender, Race, Emotion

import functions
import distance as dst



def stream(
    db_path="",
    model_name="VGG-Face",
    detector_backend="opencv",
    distance_metric="cosine",
    enable_face_analysis=True,
    source=0,
    time_threshold=5,
    frame_threshold=5,
):
    

    if time_threshold < 1:
        raise ValueError(
            "time_threshold must be greater than the value 1 but you passed " + str(time_threshold)
        )

    if frame_threshold < 1:
        raise ValueError(
            "frame_threshold must be greater than the value 1 but you passed "
            + str(frame_threshold)
        )

    realtime.analysis(
        db_path,
        model_name,
        detector_backend,
        distance_metric,
        enable_face_analysis,
        source=source,
        time_threshold=time_threshold,
        frame_threshold=frame_threshold,
    )

def build_model(model_name):
   

    # singleton design pattern
    global model_obj

    models = {
        "VGG-Face": VGGFace.loadModel,
        # "OpenFace": OpenFace.loadModel,
        # "Facenet": Facenet.loadModel,
        # "Facenet512": Facenet512.loadModel,
        # "DeepFace": FbDeepFace.loadModel,
        # "DeepID": DeepID.loadModel,
        # "Dlib": DlibWrapper.loadModel,
        # "ArcFace": ArcFace.loadModel,
        # "SFace": SFace.load_model,
        "Emotion": Emotion.loadModel,
        "Age": Age.loadModel,
        "Gender": Gender.loadModel,
        "Race": Race.loadModel,
    }

    if not "model_obj" in globals():
        model_obj = {}

    if not model_name in model_obj:
        model = models.get(model_name)
        if model:
            model = model()
            model_obj[model_name] = model
        else:
            raise ValueError(f"Invalid model_name passed - {model_name}")

    return model_obj[model_name]



def find(
    img_path,
    db_path,
    model_name="VGG-Face",
    distance_metric="cosine",
    enforce_detection=True,
    detector_backend="opencv",
    align=True,
    normalization="base",
    silent=False,
):
   

    tic = time.time()

    # -------------------------------
    if os.path.isdir(db_path) is not True:
        raise ValueError("Passed db_path does not exist!")

    target_size = functions.find_target_size(model_name=model_name)

    # ---------------------------------------

    file_name = f"representations_{model_name}.pkl"
    file_name = file_name.replace("-", "_").lower()

    if path.exists(db_path + "/" + file_name):

        if not silent:
            print(
                f"WARNING: Representations for images in {db_path} folder were previously stored"
                + f" in {file_name}. If you added new instances after the creation, then please "
                + "delete this file and call find function again. It will create it again."
            )

        with open(f"{db_path}/{file_name}", "rb") as f:
            representations = pickle.load(f)

        if not silent:
            print("There are ", len(representations), " representations found in ", file_name)

    else:  # create representation.pkl from scratch
        employees = []

        for r, _, f in os.walk(db_path):
            for file in f:
                if (
                    (".jpg" in file.lower())
                    or (".jpeg" in file.lower())
                    or (".png" in file.lower())
                ):
                    exact_path = r + "/" + file
                    employees.append(exact_path)

        if len(employees) == 0:
            raise ValueError(
                "There is no image in ",
                db_path,
                " folder! Validate .jpg or .png files exist in this path.",
            )

     

        representations = []

        # for employee in employees:
        pbar = tqdm(
            range(0, len(employees)),
            desc="Finding representations",
            disable=silent,
        )
        for index in pbar:
            employee = employees[index]

            img_objs = functions.extract_faces(
                img=employee,
                target_size=target_size,
                detector_backend=detector_backend,
                grayscale=False,
                enforce_detection=enforce_detection,
                align=align,
            )

            for img_content, _, _ in img_objs:
                embedding_obj = represent(
                    img_path=img_content,
                    model_name=model_name,
                    enforce_detection=enforce_detection,
                    detector_backend="skip",
                    align=align,
                    normalization=normalization,
                )

                img_representation = embedding_obj[0]["embedding"]

                instance = []
                instance.append(employee)
                instance.append(img_representation)
                representations.append(instance)

        # -------------------------------

        with open(f"{db_path}/{file_name}", "wb") as f:
            pickle.dump(representations, f)

        if not silent:
            print(
                f"Representations stored in {db_path}/{file_name} file."
                + "Please delete this file when you add new identities in your database."
            )

    
    
    df = pd.DataFrame(representations, columns=["identity", f"{model_name}_representation"])

    
    target_objs = functions.extract_faces(
        img=img_path,
        target_size=target_size,
        detector_backend=detector_backend,
        grayscale=False,
        enforce_detection=enforce_detection,
        align=align,
    )

    resp_obj = []

    for target_img, target_region, _ in target_objs:
        target_embedding_obj = represent(
            img_path=target_img,
            model_name=model_name,
            enforce_detection=enforce_detection,
            detector_backend="skip",
            align=align,
            normalization=normalization,
        )

        target_representation = target_embedding_obj[0]["embedding"]

        result_df = df.copy()  # df will be filtered in each img
        result_df["source_x"] = target_region["x"]
        result_df["source_y"] = target_region["y"]
        result_df["source_w"] = target_region["w"]
        result_df["source_h"] = target_region["h"]

        distances = []
        for index, instance in df.iterrows():
            source_representation = instance[f"{model_name}_representation"]

            if distance_metric == "cosine":
                distance = dst.findCosineDistance(source_representation, target_representation)
            elif distance_metric == "euclidean":
                distance = dst.findEuclideanDistance(source_representation, target_representation)
            elif distance_metric == "euclidean_l2":
                distance = dst.findEuclideanDistance(
                    dst.l2_normalize(source_representation),
                    dst.l2_normalize(target_representation),
                )
            else:
                raise ValueError(f"invalid distance metric passes - {distance_metric}")

            distances.append(distance)

            # ---------------------------

        result_df[f"{model_name}_{distance_metric}"] = distances

        threshold = dst.findThreshold(model_name, distance_metric)
        result_df = result_df.drop(columns=[f"{model_name}_representation"])
        result_df = result_df[result_df[f"{model_name}_{distance_metric}"] <= threshold]
        result_df = result_df.sort_values(
            by=[f"{model_name}_{distance_metric}"], ascending=True
        ).reset_index(drop=True)

        resp_obj.append(result_df)

    # -----------------------------------

    toc = time.time()

    if not silent:
        print("find function lasts ", toc - tic, " seconds")

    return resp_obj


def extract_faces(
    img_path,
    target_size=(224, 224),
    detector_backend="opencv",
    enforce_detection=True,
    align=True,
    grayscale=False,
):
   
    resp_objs = []
    img_objs = functions.extract_faces(
        img=img_path,
        target_size=target_size,
        detector_backend=detector_backend,
        grayscale=grayscale,
        enforce_detection=enforce_detection,
        align=align,
    )

    for img, region, confidence in img_objs:
        resp_obj = {}

        # discard expanded dimension
        if len(img.shape) == 4:
            img = img[0]

        resp_obj["face"] = img[:, :, ::-1]
        resp_obj["facial_area"] = region
        resp_obj["confidence"] = confidence
        resp_objs.append(resp_obj)

    return resp_objs



def analyze(
    img_path,
    actions=("emotion", "age", "gender", "race"),
    enforce_detection=True,
    detector_backend="opencv",
    align=True,
    silent=False,
):
    
    # ---------------------------------
    # validate actions
    if isinstance(actions, str):
        actions = (actions,)

    # check if actions is not an iterable or empty.
    if not hasattr(actions, "__getitem__") or not actions:
        raise ValueError("`actions` must be a list of strings.")

    actions = list(actions)

    # For each action, check if it is valid
    for action in actions:
        if action not in ("emotion", "age", "gender", "race"):
            raise ValueError(
                f"Invalid action passed ({repr(action)})). "
                "Valid actions are `emotion`, `age`, `gender`, `race`."
            )
    # ---------------------------------
    # build models
    models = {}
    if "emotion" in actions:
        models["emotion"] = build_model("Emotion")

    if "age" in actions:
        models["age"] = build_model("Age")

    if "gender" in actions:
        models["gender"] = build_model("Gender")

    if "race" in actions:
        models["race"] = build_model("Race")
    # ---------------------------------
    resp_objects = []

    img_objs = functions.extract_faces(
        img=img_path,
        target_size=(224, 224),
        detector_backend=detector_backend,
        grayscale=False,
        enforce_detection=enforce_detection,
        align=align,
    )

    for img_content, img_region, _ in img_objs:
        if img_content.shape[0] > 0 and img_content.shape[1] > 0:
            obj = {}
            # facial attribute analysis
            pbar = tqdm(range(0, len(actions)), desc="Finding actions", disable=silent)
            for index in pbar:
                action = actions[index]
                pbar.set_description(f"Action: {action}")

                if action == "emotion":
                    img_gray = cv2.cvtColor(img_content[0], cv2.COLOR_BGR2GRAY)
                    img_gray = cv2.resize(img_gray, (48, 48))
                    img_gray = np.expand_dims(img_gray, axis=0)

                    emotion_predictions = models["emotion"].predict(img_gray, verbose=0)[0, :]

                    sum_of_predictions = emotion_predictions.sum()

                    obj["emotion"] = {}

                    for i, emotion_label in enumerate(Emotion.labels):
                        emotion_prediction = 100 * emotion_predictions[i] / sum_of_predictions
                        obj["emotion"][emotion_label] = emotion_prediction

                    obj["dominant_emotion"] = Emotion.labels[np.argmax(emotion_predictions)]

                elif action == "age":
                    age_predictions = models["age"].predict(img_content, verbose=0)[0, :]
                    apparent_age = Age.findApparentAge(age_predictions)
                    # int cast is for exception - object of type 'float32' is not JSON serializable
                    obj["age"] = int(apparent_age)

                elif action == "gender":
                    gender_predictions = models["gender"].predict(img_content, verbose=0)[0, :]
                    obj["gender"] = {}
                    for i, gender_label in enumerate(Gender.labels):
                        gender_prediction = 100 * gender_predictions[i]
                        obj["gender"][gender_label] = gender_prediction

                    obj["dominant_gender"] = Gender.labels[np.argmax(gender_predictions)]

                elif action == "race":
                    race_predictions = models["race"].predict(img_content, verbose=0)[0, :]
                    sum_of_predictions = race_predictions.sum()

                    obj["race"] = {}
                    for i, race_label in enumerate(Race.labels):
                        race_prediction = 100 * race_predictions[i] / sum_of_predictions
                        obj["race"][race_label] = race_prediction

                    obj["dominant_race"] = Race.labels[np.argmax(race_predictions)]

                # -----------------------------
                # mention facial areas
                obj["region"] = img_region

            resp_objects.append(obj)

    return resp_objects

def represent(
    img_path,
    model_name="VGG-Face",
    enforce_detection=True,
    detector_backend="opencv",
    align=True,
    normalization="base",
):
  
    resp_objs = []

    model = build_model(model_name)

    # ---------------------------------
    # we have run pre-process in verification. so, this can be skipped if it is coming from verify.
    target_size = functions.find_target_size(model_name=model_name)
    if detector_backend != "skip":
        img_objs = functions.extract_faces(
            img=img_path,
            target_size=target_size,
            detector_backend=detector_backend,
            grayscale=False,
            enforce_detection=enforce_detection,
            align=align,
        )
    else:  # skip
        if isinstance(img_path, str):
            img = functions.load_image(img_path)
        elif type(img_path).__module__ == np.__name__:
            img = img_path.copy()
        else:
            raise ValueError(f"unexpected type for img_path - {type(img_path)}")
        # --------------------------------
        if len(img.shape) == 4:
            img = img[0]  # e.g. (1, 224, 224, 3) to (224, 224, 3)
        if len(img.shape) == 3:
            img = cv2.resize(img, target_size)
            img = np.expand_dims(img, axis=0)
        # --------------------------------
        img_region = [0, 0, img.shape[1], img.shape[0]]
        img_objs = [(img, img_region, 0)]
    # ---------------------------------

    for img, region, confidence in img_objs:
        # custom normalization
        img = functions.normalize_input(img=img, normalization=normalization)

        # represent
        if "keras" in str(type(model)):
            # new tf versions show progress bar and it is annoying
            embedding = model.predict(img, verbose=0)[0].tolist()
        else:
            # SFace and Dlib are not keras models and no verbose arguments
            embedding = model.predict(img)[0].tolist()

        resp_obj = {}
        resp_obj["embedding"] = embedding
        resp_obj["facial_area"] = region
        resp_obj["face_confidence"] = confidence
        resp_objs.append(resp_obj)

    return resp_objs

