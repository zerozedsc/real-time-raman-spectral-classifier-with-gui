from typing import Sequence
from functions.configs import *

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from collections import Counter
from datetime import datetime

import numpy as np
import ramanspy as rp
import onnxruntime as ort
import matplotlib.pyplot as plt
import skl2onnx
import onnx
import shap
import time
import sklearn
import pickle


class RamanML:
    """
    A class to handle the machine learning pipeline for Raman spectral data.

    Attributes:
    -----------
    region : tuple[int, int]
        The wavenumber region used for cropping spectral data (for reference only).
    """

    def __init__(self, region: Tuple[int, int] = (1050, 1700)):
        self.region = region
        self._model = None
        self.common_axis = None
        self.n_features_in = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X_pred = None
        self.y_pred = None

    def SVCMODEL(
        self,
        C=1.0,
        kernel='rbf',
        degree=3,
        gamma='scale',
        coef0=0.0,
        shrinking=True,
        probability=False,
        tol=0.001,
        cache_size=200,
        class_weight="balanced",
        verbose=False,
        max_iter=-1,
        decision_function_shape='ovr',
        break_ties=False,
        random_state=None
    ) -> SVC:
        """
        Set the SVC model to be used for training.

        Parameters:
            All parameters are passed to sklearn.svm.SVC.

        Returns:
            SVC
        """
        return SVC(
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            shrinking=shrinking,
            probability=probability,
            tol=tol,
            cache_size=cache_size,
            class_weight=class_weight,
            verbose=verbose,
            max_iter=max_iter,
            decision_function_shape=decision_function_shape,
            break_ties=break_ties,
            random_state=random_state
        )

    def train_svc(
        self,
        normal_data: Tuple[List[rp.SpectralContainer], str],
        disease_data: Tuple[List[rp.SpectralContainer], str],
        test_size: float = 0.3,
        param_search: bool = True,
        random_state: int = 42,
        SVC_model: SVC = SVC(kernel='rbf', C=1.0,
                             gamma='scale', class_weight='balanced')
    ) -> dict:
        """
        Train a Support Vector Machine (SVM) classifier to distinguish between normal and disease Raman spectra.

        Args:
            normal_data (Tuple[List[rp.SpectralContainer], str]): 
                Tuple containing a list of normal spectra containers and the corresponding label (e.g., 'normal').
            disease_data (Tuple[List[rp.SpectralContainer], str]): 
                Tuple containing a list of disease spectra containers and the corresponding label (e.g., 'disease').
            test_size (float, optional): 
                Proportion of the dataset to include in the test split. Defaults to 0.3.
            param_search (bool, optional): 
                If True, performs hyperparameter tuning using GridSearchCV. Defaults to True.
            random_state (int, optional): 
                Random seed for reproducibility. Defaults to 42.
            SVC_model (SVC, optional): 
                Predefined SVC model to use when param_search is False. Defaults to SVC with RBF kernel and balanced class weights.

        Returns:
            dict: Dictionary containing:
                - "confusion_matrix": Confusion matrix of predictions on the test set.
                - "classification_report": Classification report as a string.
                - "model": Trained SVM model.
                - "cross_val_score": Cross-validation scores on the training set.
                - "decision_function_score": Decision function scores on the test set.
                - "x_train": Training feature matrix.
                - "y_train": Training labels.
                - "x_test": Test feature matrix.
                - "y_test": Test labels.

        Notes:
            - All spectra are interpolated to a common wavenumber axis before training.
            - If param_search is True, hyperparameters are tuned using 10-fold stratified cross-validation.
        """
        try:
            start_time = time.time()
            normal_spectra, normal_label = normal_data
            disease_spectra, disease_label = disease_data

            # Merge all containers for each class
            def merge_containers(containers):
                if len(containers) == 1:
                    return containers[0]
                all_data = np.vstack([c.spectral_data for c in containers])
                return rp.SpectralContainer(all_data, containers[0].spectral_axis)

            normal_merged = merge_containers(normal_spectra)
            disease_merged = merge_containers(disease_spectra)

            # 1. Find the union of all wavenumbers
            normal_axis = normal_merged.spectral_axis
            disease_axis = disease_merged.spectral_axis
            common_axis = np.union1d(normal_axis, disease_axis)
            self.common_axis = common_axis
            self.n_features_in = len(common_axis)

            # 2. Interpolate all spectra to the common axis (vectorized)
            def interp_all(spectral_data, from_axis, to_axis):
                return np.array([np.interp(to_axis, from_axis, s) for s in spectral_data])

            X_normal = interp_all(normal_merged.spectral_data,
                                  normal_merged.spectral_axis, common_axis)
            X_disease = interp_all(disease_merged.spectral_data,
                                   disease_merged.spectral_axis, common_axis)
            print(
                f"Interpolated ({len(X_normal)}) {normal_label} and ({len(X_disease)}) {disease_label} spectra to common axis.")

            X = np.vstack([X_normal, X_disease])
            y = np.array([normal_label] * X_normal.shape[0] +
                         [disease_label] * X_disease.shape[0])

            # 3. Train/test split and SVM
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=random_state
            )
            print(
                f"Training SVM with {len(X_train)} training samples and {len(X_test)} test samples.")

            self.X_train = X_train
            self.y_train = y_train
            self.X_test = X_test
            self.y_test = y_test

            crossValScore = 0
            decisionFunctionScore = 0
            if param_search:
                print("Performing hyperparameter tuning with GridSearchCV...")
                param_grid = {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'gamma': ['scale', 0.01, 0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                }
                cv = StratifiedKFold(
                    n_splits=10, shuffle=True, random_state=42)
                grid = GridSearchCV(
                    SVC(class_weight='balanced'), param_grid, cv=cv)
                grid.fit(X_train, y_train)
                clf = grid.best_estimator_
                print(f"Best SVM parameters: {grid.best_params_}")
            else:
                clf = SVC_model
                cv = StratifiedKFold(
                    n_splits=10, shuffle=True, random_state=42)
                clf.fit(X_train, y_train)

            end_time = time.time() - start_time

            # 4. Cross-validation and decision function
            crossValScore = cross_val_score(clf, X_train, y_train, cv=cv)
            decisionFunctionScore = clf.decision_function(X_test)
            self._model = clf

            y_pred = clf.predict(X_test)
            conf_matrix = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred, target_names=[
                normal_label, disease_label])

            return {
                "success": True,
                "msg": "train_svc_success",
                "confusion_matrix": conf_matrix,
                "classification_report": report,
                "model": clf,
                "cross_val_score": crossValScore,
                "decision_function_score": decisionFunctionScore,
                "x_train": X_train,
                "y_train": y_train,
                "x_test": X_test,
                "y_test": y_test,
                "training_time": end_time,
            }

        except Exception as e:
            create_logs("train_svc", "ML",
                        f"Error training SVC model: {e} \n {traceback.format_exc()}", status='error')
            return {
                "success": False,
                "msg": "train_svc_error",
                "detail": f"{e} \n {traceback.format_exc()}",
            }

    def RFMODEL(
        self,
        n_estimators=100,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight="balanced",
        ccp_alpha=0.0,
        max_samples=None
    ) -> RandomForestClassifier:
        """
        Set the Random Forest model to be used for training.

        Parameters:
            All parameters are passed to sklearn.ensemble.RandomForestClassifier.
            n_estimators: int, default=100
                The number of trees in the forest.
            criterion: {"gini", "entropy"}, default="gini"
                The function to measure the quality of a split.
            max_depth: int, default=None
                The maximum depth of the tree.
            min_samples_split: int or float, default=2
                The minimum number of samples required to split an internal node.
            min_samples_leaf: int or float, default=1
                The minimum number of samples required to be at a leaf node.
            min_weight_fraction_leaf: float, default=0.0
                The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node.
            max_features: {"auto", "sqrt", "log2"}, default="sqrt"
                The number of features to consider when looking for the best split.
            max_leaf_nodes: int, default=None
                Grow trees with max_leaf_nodes in best-first fashion.
            min_impurity_decrease: float, default=0.0
                A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
            bootstrap: bool, default=True
                Whether bootstrap samples are used when building trees.
            oob_score: bool, default=False
                Whether to use out-of-bag samples to estimate the generalization accuracy.
            n_jobs: int, default=None
                The number of jobs to run in parallel. None means 1 unless in a joblib.parallel_backend context.
            random_state: int, default=None
                Controls the randomness of the estimator.
            verbose: int, default=0
                Controls the verbosity when fitting and predicting.
            warm_start: bool, default=False
                When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble.
            class_weight: dict, list of dicts, "balanced", or None, default="balanced"
                Weights associated with classes in the form {class_label: weight}.
            ccp_alpha: non-negative float, default=0.0
                Complexity parameter used for Minimal Cost-Complexity Pruning.
            max_samples: int or float, default=None
                If bootstrap is True, the number of samples to draw to train each base estimator.
                If float, it should be in (0.0, 1.0] and represent the proportion of the training set to sample.
                If int, it represents the absolute number of samples.
                If None, then draw X.shape[0] samples.
                If max_samples is not None, then bootstrap will be True.


        Returns:
            RandomForestClassifier
        """
        return RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples
        )

    def train_rf(
        self,
        normal_data: Tuple[List[rp.SpectralContainer], str],
        disease_data: Tuple[List[rp.SpectralContainer], str],
        test_size: float = 0.3,
        param_search: bool = True,
        random_state: int = 42,
        RF_model: RandomForestClassifier = RandomForestClassifier(
            class_weight='balanced', random_state=42)
    ) -> dict:
        """
        Train a Random Forest model on Raman data (normal vs. disease).

        Attributes:
            normal_data (Tuple[List[rp.SpectralContainer], str]): List of normal spectra and label string (e.g., 'normal')
            disease_data (Tuple[List[rp.SpectralContainer], str]): List of disease spectra and label string (e.g., 'disease')      
            test_size (float): Train/test split ratio           
            param_search (bool): Whether to use GridSearchCV to tune RF hyperparameters           
            random_state (int): Random state for reproducibility           
            RF_model (RandomForestClassifier): A predefined RandomForestClassifier model to use when param_search is False

        Returns:
            dict: Dictionary containing:
                - "confusion_matrix": Confusion matrix of predictions on the test set.
                - "classification_report": Classification report as a string.
                - "model": Trained Random Forest model.
                - "cross_val_score": Cross-validation scores on the training set.
                - "feature_importances": Feature importances from the trained model.
                - "x_train": Training feature matrix.
                - "y_train": Training labels.
                - "x_test": Test feature matrix.
                - "y_test": Test labels.
                - "training_time": Time taken in seconds to train the model.
        """

        try:
            start_time = time.time()
            normal_spectra, normal_label = normal_data
            disease_spectra, disease_label = disease_data

            def merge_containers(containers):
                if len(containers) == 1:
                    return containers[0]
                all_data = np.vstack([c.spectral_data for c in containers])
                return rp.SpectralContainer(all_data, containers[0].spectral_axis)

            normal_merged = merge_containers(normal_spectra)
            disease_merged = merge_containers(disease_spectra)

            normal_axis = normal_merged.spectral_axis
            disease_axis = disease_merged.spectral_axis
            common_axis = np.union1d(normal_axis, disease_axis)
            self.common_axis = common_axis
            self.n_features_in = len(common_axis)

            def interp_all(spectral_data, from_axis, to_axis):
                return np.array([np.interp(to_axis, from_axis, s) for s in spectral_data])

            X_normal = interp_all(normal_merged.spectral_data,
                                  normal_merged.spectral_axis, common_axis)
            X_disease = interp_all(disease_merged.spectral_data,
                                   disease_merged.spectral_axis, common_axis)
            print(
                f"Interpolated ({len(X_normal)}) {normal_label} and ({len(X_disease)}) {disease_label} spectra to common axis.")

            X = np.vstack([X_normal, X_disease])
            y = np.array([normal_label] * X_normal.shape[0] +
                         [disease_label] * X_disease.shape[0])

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=random_state
            )

            self.X_train = X_train
            self.y_train = y_train
            self.X_test = X_test
            self.y_test = y_test

            print(
                f"Training Random Forest with {len(X_train)} training samples and {len(X_test)} test samples.")
            crossValScore = 0
            featureImportances = 0
            if param_search:
                print("Performing hyperparameter tuning with GridSearchCV...")
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 5, 10, 20],
                    'min_samples_split': [2, 5, 10],
                }
                cv = StratifiedKFold(
                    n_splits=10, shuffle=True, random_state=42)
                grid = GridSearchCV(RandomForestClassifier(
                    class_weight='balanced', random_state=42), param_grid, cv=cv)
                grid.fit(X_train, y_train)
                clf = grid.best_estimator_
                print(f"Best RF parameters: {grid.best_params_}")
            else:
                clf = RF_model
                cv = StratifiedKFold(
                    n_splits=10, shuffle=True, random_state=42)
                clf.fit(X_train, y_train)

            end_time = time.time() - start_time
            crossValScore = cross_val_score(clf, X_train, y_train, cv=cv)
            featureImportances = clf.feature_importances_
            self._model = clf

            y_pred = clf.predict(X_test)
            conf_matrix = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred, target_names=[
                normal_label, disease_label])

            return {
                "success": True,
                "msg": "train_rf_success",
                "confusion_matrix": conf_matrix,
                "classification_report": report,
                "model": clf,
                "cross_val_score": crossValScore,
                "feature_importances": featureImportances,
                "x_train": X_train,
                "y_train": y_train,
                "x_test": X_test,
                "y_test": y_test,
                "training_time": end_time
            }

        except Exception as e:
            create_logs("train_rf", "ML",
                        f"Error training RF model: {e} \n {traceback.format_exc()}", status='error')
            return {
                "success": False,
                "msg": "train_rf_error",
                "detail": f"{e} \n {traceback.format_exc()}",
            }

    def predict(self, test_spectra: List[rp.SpectralContainer], model: Union[SVC, RandomForestClassifier, None] = None, common_axis: np.union1d = None) -> dict:
        """
        Predict class labels using a trained ML model (SVC, RF, etc.).

        Parameters:
            test_spectra (List[rp.SpectralContainer]):
                List of Raman spectral containers to classify.
            model (scikit-learn classifier or None):
                Optional model to override the stored one.

        Returns:
            dict with prediction results:
            {
                "label_percentages": dict of label percentages,
                "most_common_label": str of the most common label,
                "y_pred": np.ndarray of predicted labels,
            }
        """
        # Determine which model to use
        clf = model

        if clf is None:
            if hasattr(self, "_model") and self._model is not None:
                clf = self._model
            else:
                raise ValueError(
                    "No trained model found. Train a model first or provide one.")

        if common_axis is None and hasattr(self, "common_axis") and self.common_axis is not None:
            common_axis = getattr(self, "common_axis", None)
        else:
            raise ValueError("No common axis found for the model.")

        # Determine the reference axis
        if hasattr(clf, "support_vectors_"):
            n_features = clf.support_vectors_.shape[1]
        elif hasattr(clf, "feature_importances_"):
            n_features = clf.feature_importances_.shape[0]
        else:
            raise ValueError("No reference axis found for the provided model.")

        # Prepare test data
        X_test = []
        for s in test_spectra:
            if s.spectral_data.shape[1] != n_features:
                for spec in s.spectral_data:
                    interp = np.interp(common_axis, s.spectral_axis, spec)
                    X_test.append(interp)
            else:
                for spec in s.spectral_data:
                    X_test.append(spec)
        X_test = np.array(X_test)
        self.X_pred = X_test

        print(
            f"Predicting {len(X_test)} test samples with {n_features} features.")

        y_pred = clf.predict(X_test)
        if hasattr(clf, "predict_proba"):
            try:
                proba = clf.predict_proba(X_test)
                class_labels = clf.classes_
            except Exception as e:
                print(f"Model does not support probability estimates \n{e}")
                proba = None
                class_labels = None
        else:
            proba = None
            class_labels = None

        self.y_pred = y_pred
        results = []
        for idx, pred in enumerate(y_pred):
            entry = {"predicted_label": pred}
            if proba is not None and class_labels is not None:
                prob_dict = {str(class_labels[i]): float(
                    proba[idx][i]) for i in range(len(class_labels))}
                entry["confidence"] = float(prob_dict[str(pred)])
                entry["all_probabilities"] = prob_dict
            else:
                entry["confidence"] = None
                entry["all_probabilities"] = None
            results.append(entry)

        labels = [res['predicted_label'] for res in results]
        label_counts = Counter(labels)
        total = len(labels)
        label_percentages = {label: count /
                             total for label, count in label_counts.items()}

        return {
            "label_percentages": label_percentages,
            "most_common_label": max(label_counts, key=label_counts.get),
            "y_pred": y_pred,
            # <-- add this line
            "confidences": [res["confidence"] for res in results],
            # <-- add this line
            "all_probabilities": [res["all_probabilities"] for res in results],
        }

    def predict_with_threshold(
        self,
        test_spectra: list[rp.SpectralContainer],
        positive_label: str,
        threshold: float = 0.7,
        model: RandomForestClassifier = None
    ) -> dict:
        """
        Predict using a custom probability threshold for the positive class label. (Only for RF models)

        Args:
            test_spectra: List of SpectralContainer to classify.
            positive_label: The label string for the positive (disease) class.
            threshold: Probability threshold for positive prediction.
            model: Optionally override the stored model.

        Returns:
            dict: Same as predict(), but with thresholding applied.
        """
        clf = model or getattr(self, "_model", None)
        if clf is None and clf is not RandomForestClassifier:
            raise ValueError("No trained RandomForest model found.")

        # Use the same axis logic as in predict()
        common_axis = getattr(self, "common_axis", None)
        n_features = getattr(self, "n_features_in", None)
        if common_axis is None or n_features is None:
            raise ValueError("No reference axis found for the model.")

        # Prepare test data
        X_test = []
        for s in test_spectra:
            if s.spectral_data.shape[1] != n_features:
                for spec in s.spectral_data:
                    interp = np.interp(common_axis, s.spectral_axis, spec)
                    X_test.append(interp)
            else:
                for spec in s.spectral_data:
                    X_test.append(spec)
        X_test = np.array(X_test)
        self.X_pred = X_test

        # Predict probabilities
        proba = clf.predict_proba(X_test)
        class_labels = clf.classes_
        pos_idx = np.where(class_labels == positive_label)[0][0]

        y_pred = []
        confidences = []
        all_probabilities = []
        for i, p in enumerate(proba):
            prob_pos = p[pos_idx]
            if prob_pos >= threshold:
                pred = positive_label
            else:
                # Pick the other label (assume binary)
                pred = [l for l in class_labels if l != positive_label][0]
            y_pred.append(pred)
            confidences.append(prob_pos)
            all_probabilities.append(
                {str(class_labels[j]): float(p[j]) for j in range(len(class_labels))})

        self.y_pred = y_pred
        label_counts = Counter(y_pred)
        total = len(y_pred)
        label_percentages = {label: count /
                             total for label, count in label_counts.items()}

        return {
            "label_percentages": label_percentages,
            "most_common_label": max(label_counts, key=label_counts.get),
            "y_pred": np.array(y_pred),
            "confidences": confidences,
            "all_probabilities": all_probabilities,
        }


class MLModel:
    def __init__(self, onnx_path: str = None, meta_path: str = None, pickle_path: str = None,
                 sess_options: Any | None = None,
                 providers: Sequence[str | tuple[str,
                                                 dict[Any, Any]]] | None = None,
                 provider_options: Sequence[dict[Any,
                                                 Any]] | None = None,
                 **kwargs: Any):
        self.session = None
        self.metadata = None
        self.common_axis = None
        self.n_features_in = None
        self.onnx_model = None
        self.sklearn_model = None
        self.load_msg = None
        self.load_success = False
        load_data = {}

        if onnx_path and meta_path:
            load_data = self.load(onnx_path, meta_path, sess_options,
                                  providers, provider_options, **kwargs)
            self.load_msg = load_data.get("msg", None)
            self.load_success = load_data.get("success", False)

        if pickle_path:
            pickle_data = self.pickle_load(pickle_path, meta_path)
            if pickle_data.get("success", False):
                self.sklearn_model = pickle_data.get("model", None)
                self.metadata = pickle_data.get("metadata", None)
                self.common_axis = np.array(self.metadata["common_axis"])
                self.n_features_in = int(self.metadata["n_features_in"])
            else:
                create_logs("pickle_load", "ML",
                            f"Error loading pickle model: {pickle_data.get('detail', '')}", status='error')

    def save(self, model: Union[SVC, RandomForestClassifier], labels: list[str], filename: str,
             common_axis: np.ndarray, n_features_in: int, save_pickle: bool = True,
             meta: dict = {
        "model_type": "", "model_name": "",
        "model_version": "", "model_description": "",
        "model_author": "", "model_date": "",
            "model_license": "", "model_source": ""}, other_meta: dict = {}) -> dict:
        """
        Save the trained SVC model to ONNX format.

        Parameters:
            mlresult (dict): Dictionary containing the trained model and other results.
        """

        try:
            n_features = model.n_features_in_  # or use your input shape

            # Ensure the models directory exists
            model_dir = os.path.join(CURRENT_DIR, "models")
            os.makedirs(model_dir, exist_ok=True)

            filename_safe = safe_filename(filename)
            onnx_path = os.path.join(model_dir, f"{filename_safe}.onnx")
            meta_path = os.path.join(model_dir, f"{filename_safe}.json")
            pickle_path = os.path.join(model_dir, f"{filename_safe}.pkl")

            initial_type = [
                ('float_input', FloatTensorType([None, n_features]))]
            onnx_model = convert_sklearn(model, initial_types=initial_type)

            # Save the model to ONNX format
            with open(onnx_path, "wb") as f:
                f.write(onnx_model.SerializeToString())

            # Save the model to a pickle file
            if save_pickle:
                with open(pickle_path, "wb") as f:
                    pickle.dump(model, f)

            # Save metadata
            meta = meta or {}
            metadata = {
                "model_type": meta.get("model_type", type(model).__name__),
                "model_name": meta.get("model_name", ""),
                "model_version": meta.get("model_version", "1.0.0"),
                "model_description": meta.get("model_description", ""),
                "model_author": meta.get("model_author", ""),
                "model_date": meta.get("model_date", datetime.now().strftime("%Y-%m-%d")),
                "model_license": meta.get("model_license", ""),
                "model_source": meta.get("model_source", ""),
                "labels": labels,
                "common_axis": common_axis.tolist(),
                "n_features_in": int(n_features_in),
            }

            # Add all sklearn model parameters
            if hasattr(model, "get_params"):
                metadata["model_params"] = model.get_params()
            else:
                metadata["model_params"] = {}

            metadata["library"] = {
                "sklearn": sklearn.__version__,
                "onnx": skl2onnx.__version__,
                "onnxruntime": ort.__version__,
                "numpy": np.__version__,
                "skl2onnx": skl2onnx.__version__,
            }

            metadata.update(other_meta)

            # save metadata to JSON
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)

            save_str = f"(SAVED SUCCESS) onnx: {onnx_path} , metadata: {meta_path}"
            if save_pickle:
                save_str += f", pkl: {pickle_path}"

            create_logs("save_onnx", "ML",
                        f"Model saved to {onnx_path} and metadata to {meta_path}", status='info')

            self.onnx_model = onnx_model
            self.metadata = metadata
            self.common_axis = common_axis
            self.n_features_in = int(n_features_in)
            self.session = ort.InferenceSession(
                onnx_path, providers=["CPUExecutionProvider"])

            return {
                "onnx_model": onnx_model,
                "onnx_path": onnx_path,
                "meta_path": meta_path,
                "metadata": metadata,
                "success": True,
            }

        except Exception as e:
            create_logs("save_onnx", "ML",
                        f"Error saving model to ONNX: {e}", status='error')
            return {
                "success": False,
                "msg": "save_onnx_error",
                "detail": f"{e} \n {traceback.format_exc()}",
            }

    def load(self, onnx_path: str = None, meta_path: str = None, sess_options: Any | None = None,
             providers: Sequence[str | tuple[str,
                                             dict[Any, Any]]] | None = None,
             provider_options: Sequence[dict[Any,
                                             Any]] | None = None,
             **kwargs: Any) -> dict:
        """
        Load an ONNX model and (optionally) its metadata, and create an ONNX Runtime inference session.

        Parameters:
            onnx_path (str): Path to the ONNX model file.
            meta_path (str, optional): Path to the metadata JSON file.

        Returns:
            dict: {
                "onnx_session": ONNX Runtime inference session,
                "metadata": dict (if meta_path is provided),
                "onnx_model": loaded ONNX model,
                "success": True/False,
                "msg": status message,
            }
        """
        try:
            # Load ONNX model
            onnx_model = onnx.load(onnx_path)
            # Create ONNX Runtime inference session
            session = ort.InferenceSession(onnx_path,
                                           sess_options=sess_options,
                                           providers=providers,
                                           provider_options=provider_options,
                                           **kwargs)

            # Load metadata if provided
            metadata = None
            if meta_path and os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    metadata = json.load(f)

            create_logs("load_onnx", "ML",
                        f"Model loaded from {onnx_path} and metadata from {meta_path}", status='info')

            self.onnx_model = onnx_model
            self.metadata = metadata
            self.common_axis = np.array(metadata["common_axis"])
            self.n_features_in = int(metadata["n_features_in"])
            self.session = session
            self.load_msg = "load_onnx_success"
            self.load_success = True

            return {
                "onnx_session": session,
                "metadata": metadata,
                "onnx_model": onnx_model,
                "success": True,
                "msg": "load_onnx_success",
            }
        except Exception as e:
            create_logs("load_onnx", "ML",
                        f"Error loading ONNX model: {e} \n {traceback.format_exc()}", status='error')
            self.load_msg = "load_onnx_error"
            self.load_success = False
            return {
                "success": False,
                "msg": "load_onnx_error",
                "detail": f"{e} \n {traceback.format_exc()}",
            }

    def pickle_load(self, pickle_path: str, meta_path: str) -> dict:
        """
        Load a pickled model and its metadata.

        Args:
            pickle_path (str): Path to the pickled model file.
            meta_path (str): Path to the metadata JSON file.

        Returns:
            dict: {
                "model": loaded model,
                "metadata": dict,
                "success": True/False,
                "msg": status message,
            }
        """
        try:
            model = None
            with open(pickle_path, "rb") as f:
                model = pickle.load(f)

            self.sklearn_model = model

            with open(meta_path, "r") as f:
                metadata = json.load(f)

            self.metadata = metadata
            self.common_axis = np.array(metadata["common_axis"])
            self.n_features_in = int(metadata["n_features_in"])

            create_logs("pickle_load", "ML",
                        f"Model loaded from {pickle_path} and metadata from {meta_path}", status='info')

            return {
                "model": model,
                "metadata": metadata,
                "success": True,
                "msg": "pickle_load_success",
            }
        except Exception as e:
            create_logs("pickle_load", "ML",
                        f"Error loading pickled model: {e} \n {traceback.format_exc()}", status='error')
            return {
                "success": False,
                "msg": "pickle_load_error",
                "detail": f"{e} \n {traceback.format_exc()}",
            }

    def predict(
        self,
        test_spectra: list,
        session: ort.InferenceSession = None,
        n_features: int = None,
        common_axis: np.ndarray = None,
        class_labels: list = None
    ) -> dict:
        """
        Predict class labels using an ONNX Runtime session, mimicking RamanML.predict.

        Args:
            test_spectra (list): List of Raman spectral containers to classify.
            session (ort.InferenceSession, optional): ONNX Runtime session. If None, uses the stored session.
            n_features (int, optional): Number of features in the input data. If None, uses the stored value or metadata.
            common_axis (np.ndarray, optional): Common axis for interpolation. If None, uses the stored value or metadata.
            class_labels (list, optional): List of class labels. If None, tries to extract from the session.

        Returns:
            dict: {
            "label_percentages": dict of label percentages,
            "most_common_label": str of the most common label,
            "y_pred": np.ndarray of predicted labels,
            "confidences": list of confidence scores,
            "all_probabilities": list of all probabilities for each class,
            }

        """
        # Prefer explicit arguments, then self, then metadata
        if n_features is None:
            if hasattr(self, "n_features_in") and self.n_features_in is not None:
                n_features = self.n_features_in
            elif self.metadata and "n_features_in" in self.metadata:
                n_features = self.metadata["n_features_in"]
            else:
                raise ValueError(
                    "n_features is not provided and not found in self or metadata.")

        if common_axis is None:
            if hasattr(self, "common_axis") and self.common_axis is not None:
                common_axis = self.common_axis
            elif self.metadata and "common_axis" in self.metadata:
                common_axis = np.array(self.metadata["common_axis"])
            else:
                raise ValueError(
                    "common_axis is not provided and not found in self or metadata.")

        # Prepare test data (interpolate if needed)
        X_test = []
        for s in test_spectra:
            if s.spectral_data.shape[1] != n_features:
                for spec in s.spectral_data:
                    interp = np.interp(common_axis, s.spectral_axis, spec)
                    X_test.append(interp)
            else:
                for spec in s.spectral_data:
                    X_test.append(spec)
        X_test = np.array(X_test, dtype=np.float32)

        # ONNX inference
        session = session if session else self.session
        if session is None:
            create_logs("onnx_predict", "ML",
                        "No ONNX session found. Load a model first or provide one.", status='error')
            raise ValueError(
                "No ONNX session found. Load a model first or provide one.")

        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: X_test})

        # Try to find label and probability outputs
        if len(outputs) == 1:
            y_pred = outputs[0]
            proba = None
        elif len(outputs) == 2:
            # Try to detect which output is which based on shape and type
            if outputs[0].ndim > 1 and outputs[0].shape[1] > 1:
                # First output has multiple columns - likely probabilities
                proba, y_pred = outputs[0], outputs[1]
            else:
                # Otherwise assume standard order
                y_pred, proba = outputs[0], outputs[1]
        else:
            y_pred = outputs[0]
            proba = None

        # Make sure proba is numeric
        if proba is not None:
            if isinstance(proba, list):
                proba = np.array(proba)
            if not np.issubdtype(proba.dtype, np.number):
                try:
                    proba = proba.astype(np.float32)
                except Exception:
                    create_logs("onnx_predict", "ML",
                                f"Warning: Could not convert probabilities to float: {getattr(proba, 'dtype', type(proba))}", "WARNING")
                    proba = None
        # Convert bytes to str if needed
        if hasattr(y_pred, "dtype") and y_pred.dtype.kind in {"S", "O"}:
            y_pred = np.array(
                [x.decode() if isinstance(x, bytes) else x for x in y_pred])

        results = []
        for idx, pred in enumerate(y_pred):
            entry = {"predicted_label": pred}
            if proba is not None:
                # Get class labels from session if not provided
                if class_labels is None:
                    try:
                        class_labels = session.get_outputs()[1].type.shape[1]
                    except Exception:
                        class_labels = [str(i) for i in range(proba.shape[1])]
                if class_labels is None:
                    class_labels = [str(i) for i in range(proba.shape[1])]
                prob_dict = {str(class_labels[i]): float(
                    proba[idx][i]) for i in range(len(class_labels))}
                entry["confidence"] = float(
                    prob_dict.get(str(pred), np.max(proba[idx])))
                entry["all_probabilities"] = prob_dict
            else:
                entry["confidence"] = None
                entry["all_probabilities"] = None
            results.append(entry)

        labels = [res['predicted_label'] for res in results]
        label_counts = Counter(labels)
        total = len(labels)
        label_percentages = {label: count /
                             total for label, count in label_counts.items()}

        return {
            "label_percentages": label_percentages,
            "most_common_label": max(label_counts, key=label_counts.get),
            "y_pred": y_pred,
            "confidences": [res["confidence"] for res in results],
            "all_probabilities": [res["all_probabilities"] for res in results],
        }

    def predict_numpy(self, X: np.ndarray, class_labels: list = None):
        """
        Predict class labels for a numpy array input (used for decision boundary grid).
        """
        session = self.session
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: X.astype(np.float32)})
        if len(outputs) == 1:
            y_pred = outputs[0]
        elif len(outputs) == 2:
            # Try to detect which output is which based on shape and type
            if outputs[0].ndim > 1 and outputs[0].shape[1] > 1:
                # First output has multiple columns - likely probabilities
                proba, y_pred = outputs[0], outputs[1]
            else:
                y_pred, proba = outputs[0], outputs[1]
        else:
            y_pred = outputs[0]
        # Convert bytes to str if needed
        if hasattr(y_pred, "dtype") and y_pred.dtype.kind in {"S", "O"}:
            y_pred = np.array(
                [x.decode() if isinstance(x, bytes) else x for x in y_pred])
        return y_pred
