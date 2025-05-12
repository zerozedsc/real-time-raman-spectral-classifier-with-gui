from functions.configs import *

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

from skl2onnx import convert_sklearn, to_sklearn
from skl2onnx.common.data_types import FloatTensorType

from collections import Counter
from datetime import datetime

import numpy as np
import ramanspy as rp
import skl2onnx
import onnx
import shap
import time


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
        self.clf_model = None
        self.svc_common_axis = None

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
            self.svc_common_axis = common_axis

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
            self.clf_model = clf

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
            dict: A dictionary with confusion matrix, classification report, and model
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
            self.rf_common_axis = common_axis

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
            self.rf_model = clf

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

    def predict(self, test_spectra: List[rp.SpectralContainer], model: Union[SVC, RandomForestClassifier, None] = None) -> dict:
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
            if hasattr(self, "clf_model") and self.clf_model is not None:
                clf = self.clf_model
                common_axis = getattr(self, "svc_common_axis", None)
            elif hasattr(self, "rf_model") and self.rf_model is not None:
                clf = self.rf_model
                common_axis = getattr(self, "rf_common_axis", None)
            else:
                raise ValueError(
                    "No trained model found. Train a model first or provide one.")

        # Determine the reference axis
        if hasattr(self, "svc_common_axis") and self.svc_common_axis is not None and hasattr(clf, "support_vectors_"):
            common_axis = self.svc_common_axis
            n_features = clf.support_vectors_.shape[1]
        elif hasattr(self, "rf_common_axis") and self.rf_common_axis is not None and hasattr(clf, "feature_importances_"):
            common_axis = self.rf_common_axis
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

    def shap_explain(
        self,
        model: Union[SVC, RandomForestClassifier, None] = None,
        background_data: np.ndarray = None,
        test_data: np.ndarray = None,
        nsamples: int = 100
    ) -> dict:
        """
        Generate SHAP explanations for the given model and data.

        Args:
            model (sklearn classifier or None): The trained model to explain. If None, uses the stored model.
            background_data (np.ndarray): Background data for SHAP explainer (usually a subset of training data).
            test_data (np.ndarray): Data to explain (e.g., test set).
            nsamples (int): Number of samples for SHAP estimation (default: 100).

        Returns:
            dict: {
                "shap_values": SHAP values array,
                "expected_value": SHAP expected value,
                "explainer": SHAP explainer object,
                "summary_plot": matplotlib figure object (summary plot),
            }
        """
        if model is None:
            if hasattr(self, "clf_model") and self.clf_model is not None:
                model = self.clf_model
            elif hasattr(self, "rf_model") and self.rf_model is not None:
                model = self.rf_model
            else:
                raise ValueError(
                    "No trained model found. Train a model first or provide one.")

        if background_data is None or test_data is None:
            raise ValueError(
                "background_data and test_data must be provided as numpy arrays.")

        # Choose SHAP explainer based on model type
        if isinstance(model, RandomForestClassifier):
            explainer = shap.TreeExplainer(model, background_data)
        elif isinstance(model, SVC):
            # For SVC, use KernelExplainer (slower)
            explainer = shap.KernelExplainer(
                model.predict_proba, background_data)
        else:
            raise ValueError("Unsupported model type for SHAP explanation.")

        # Compute SHAP values
        shap_values = explainer.shap_values(test_data, nsamples=nsamples)
        expected_value = explainer.expected_value

        # Generate summary plot (returns matplotlib figure)
        summary_plot = shap.summary_plot(shap_values, test_data, show=False)
        fig = shap.plt.gcf()

        return {
            "shap_values": shap_values,
            "expected_value": expected_value,
            "explainer": explainer,
            "summary_plot": fig,
        }


# Suppose your model is mlresult["model"]
def save_model_to_onnx(model: Any[SVC, RandomForestClassifier, ], labels: list[str], filename: str, meta: dict = {
    "model_type": "", "model_name": "",
    "model_version": "", "model_description": "",
    "model_author": "", "model_date": "",
        "model_license": "", "model_source": "", }) -> dict:
    """
    Save the trained SVC model to ONNX format.

    Parameters:
        mlresult (dict): Dictionary containing the trained model and other results.
    """

    try:
        n_features = model.n_features_in_  # or use your input shape

        initial_type = [('float_input', FloatTensorType([None, n_features]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)

        filename = safe_filename(filename)

        with open(f"model/{filename}.onnx", "wb") as f:
            f.write(onnx_model.SerializeToString())

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
        }

        # Add all sklearn model parameters
        if hasattr(model, "get_params"):
            metadata["model_params"] = model.get_params()
        else:
            metadata["model_params"] = {}

        with open(f"model/{filename}.json", "w") as f:
            json.dump(metadata, f, indent=2)

        create_logs("save_onnx", "ML",
                    f"Model saved to {filename}.onnx and metadata to {filename}.json", status='info')
        return {
            "onnx_model": onnx_model,
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

    # Convert the model to ONNX format


def load_model_from_onnx(onnx_path: str, meta_path: str = None) -> dict:
    """
    Load an ONNX model and (optionally) its metadata, and convert it to a usable sklearn-like model.

    Parameters:
        onnx_path (str): Path to the ONNX model file.
        meta_path (str, optional): Path to the metadata JSON file.

    Returns:
        dict: {
            "sklearn_model": sklearn-API compatible model (ONNXRuntimeInferencePipeline),
            "metadata": dict (if meta_path is provided),
            "onnx_model": loaded ONNX model,
        }
    """
    try:
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        # Convert ONNX model to sklearn-API compatible pipeline
        sklearn_model = to_sklearn(onnx_model)

        # Load metadata if provided
        metadata = None
        if meta_path and os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                metadata = json.load(f)

        create_logs("load_onnx", "ML",
                    f"Model loaded from {onnx_path} and metadata from {meta_path}", status='info')

        return {
            "sklearn_model": sklearn_model,
            "metadata": metadata,
            "onnx_model": onnx_model,
            "success": True,
            "msg": "load_onnx_success",
        }
    except Exception as e:
        create_logs("load_onnx", "ML",
                    f"Error loading ONNX model: {e} \n {traceback.format_exc()}", status='error')
        return {
            "success": False,
            "msg": "load_onnx_error",
            "detail": f"{e} \n {traceback.format_exc()}",
        }
