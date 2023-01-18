import torch_explain as te
from sklearn.metrics import f1_score
from torch_explain.logic.metrics import complexity
import torch.nn.functional as F
import copy
from typing import List

import torch
from sympy import simplify_logic, to_dnf, lambdify

from torch_explain.logic.utils import replace_names
from torch_explain.nn import Conceptizator
from torch_explain.nn.logic import EntropyLinear
from data_utils import reshape_graph_to_node_data, relabel_batch

def explain_classes(model, concepts, y, train_mask, test_mask, max_minterm_complexity=1000, topk_explanations=1000, batch=None, train_batch=None):

    if train_batch is not None:
        expanded_y = reshape_graph_to_node_data(y, batch)
        expanded_train_mask = reshape_graph_to_node_data(train_mask, batch)
        expanded_test_mask = reshape_graph_to_node_data(test_mask, batch)

        y1h = F.one_hot(y[train_mask])
        expanded_y1h = F.one_hot(expanded_y[expanded_train_mask])

        y1h_test = F.one_hot(y[test_mask])
        expanded_y1h_test = F.one_hot(expanded_y[expanded_test_mask])
    else:
        y1h = F.one_hot(y[train_mask])
        y1h_test = F.one_hot(y[test_mask])

    explanations = {}

    for class_id in range(y1h.shape[1]):
        if train_batch is not None:
            explanation, _ = explain_class(model.lens, concepts[expanded_train_mask], y1h,
                                                   concepts[expanded_train_mask], y1h, target_class=class_id,
                                                   max_minterm_complexity=max_minterm_complexity, topk_explanations=topk_explanations,
                                                   batch=train_batch, expanded_y1h=expanded_y1h, expanded_y_val1h=expanded_y1h)

            explanation_accuracy, _ = test_explanation(explanation, concepts[expanded_test_mask], y1h_test, target_class=class_id, expanded_y=expanded_y1h_test)
        else:
            explanation, _ = explain_class(model.lens, concepts[train_mask], y1h,
                                                   concepts[train_mask], y1h, target_class=class_id,
                                                   max_minterm_complexity=max_minterm_complexity, topk_explanations=topk_explanations)

            explanation_accuracy, _ = test_explanation(explanation, concepts[test_mask], y1h_test, target_class=class_id)

        explanation_complexity = complexity(explanation)

        explanations[str(class_id)] = {'explanation': explanation,
                                        'explanation_accuracy': explanation_accuracy,
                                        'explanation_complexity': explanation_complexity}

        print(f'Explanation class {class_id}: {explanation} - acc. = {explanation_accuracy:.4f} - compl. = {explanation_complexity:.4f}')

    return explanations





def explain_class(model: torch.nn.Module, x, y1h, x_val: torch.Tensor, y_val1h: torch.Tensor,
                  target_class: int, max_minterm_complexity: int = None, topk_explanations: int = 3,
                  max_accuracy: bool = False, concept_names: List = None, batch: torch.Tensor = None, expanded_y1h=None, expanded_y_val1h=None) -> [str, str]:
    """
    Generate a local explanation for a single sample.

    :param model: pytorch model
    :param x: input samples to extract logic formulas.
    :param y1h: target labels to extract logic formulas (MUST be one-hot encoded).
    :param x_val: input samples to validate logic formulas.
    :param y_val1h: target labels to validate logic formulas (MUST be one-hot encoded).
    :param target_class: target class.
    :param max_minterm_complexity: maximum number of concepts per logic formula (per sample).
    :param topk_explanations: number of local explanations to be combined.
    :param max_accuracy: if True a formula is simplified only if the simplified formula gets 100% accuracy.
    :param concept_names: list containing the names of the input concepts.
    :return: Global explanation
    """
    if batch is not None:
        x_correct, y_correct1h, expanded_y_correct1h = _get_correct_data_pooled(x, y1h, model, target_class, batch, expanded_y1h)
    else:
        x_correct, y_correct1h = _get_correct_data(x, y1h, model, target_class)

    if x_correct is None:
        return None, None

    activation = 'identity_bool'
    feature_names = [f'feature{j:010}' for j in range(x_correct.size(-1))]
    conceptizator = Conceptizator(activation)
    y_correct = conceptizator(y_correct1h[:, target_class])
    y_val = conceptizator(y_val1h[:, target_class])

    class_explanation = ''
    class_explanation_raw = ''
    for layer_id, module in enumerate(model.children()):
        if module.__class__.__name__ == 'EntropyLinear':
            local_explanations = []
            local_explanations_accuracies = {}
            local_explanations_raw = {}

            # look at the "positive" rows of the truth table only
            positive_samples = torch.nonzero(y_correct)
            for positive_sample in positive_samples:
                local_explanation, local_explanation_raw = _local_explanation(module, feature_names, positive_sample,
                                                                              local_explanations_raw,
                                                                              x_correct, y_correct1h,
                                                                              target_class, max_accuracy,
                                                                              max_minterm_complexity, expanded_y_correct1h)
                # test explanation accuracy
                if local_explanation_raw not in local_explanations_accuracies:
                    accuracy, _ = test_explanation(local_explanation_raw, x_val, y_val1h, target_class, expanded_y=expanded_y_val1h)
                    local_explanations_accuracies[local_explanation_raw] = (local_explanation, accuracy)

                if local_explanation and local_explanation_raw:
                    local_explanations_raw[local_explanation_raw] = local_explanation_raw
                    local_explanations.append(local_explanation)



            # aggregate local explanations and replace concept names in the final formula
            aggregated_explanation, best_acc = _aggregate_explanations(local_explanations_accuracies,
                                                                       topk_explanations,
                                                                       target_class, x_val, y_val1h, expanded_y=expanded_y_val1h)
            class_explanation_raw = str(aggregated_explanation)
            class_explanation = class_explanation_raw
            if concept_names is not None:
                class_explanation = replace_names(class_explanation, concept_names)

            break

    return class_explanation[1:-1], class_explanation_raw


# def test_explanation(formula: str, x: torch.Tensor, y: torch.Tensor, target_class: int):
#     """
#     Tests a logic formula.
#
#     :param formula: logic formula
#     :param x: input data
#     :param y: input labels (MUST be one-hot encoded)
#     :param target_class: target class
#     :return: Accuracy of the explanation and predictions
#     """
#
#     if formula in ['True', 'False', ''] or formula is None:
#         return 0.0, None
#
#     else:
#         assert len(y.shape) == 2
#         y = y[:, target_class]
#         concept_list = [f"feature{i:010}_{j:010}" for i in range(x.shape[2]) for j in range(x.shape[1])]
#         x = x.reshape(x.shape[0], -1)
#         # get predictions using sympy
#         explanation = to_dnf(formula)
#         fun = lambdify(concept_list, explanation, 'numpy')
#         x = x.cpu().detach().numpy()
#         predictions = fun(*[x[:, i] > 0.5 for i in range(x.shape[1])])
#         # get accuracy
#         accuracy = f1_score(y, predictions, average='macro')
#         return accuracy, predictions

def test_explanation(formula: str, x: torch.Tensor, y: torch.Tensor, target_class: int, expanded_y=None):
    """
    Tests a logic formula.
    :param formula: logic formula
    :param x: input data
    :param y: input labels (MUST be one-hot encoded)
    :param target_class: target class
    :return: Accuracy of the explanation and predictions
    """
    if formula in ['True', 'False', ''] or formula is None:
        return 0.0, None

    else:
        assert len(y.shape) == 2
        y = y[:, target_class]
        concept_list = [f"feature{i:010}" for i in range(x.shape[1])]
        # get predictions using sympy
        explanation = to_dnf(formula)
        fun = lambdify(concept_list, explanation, 'numpy')
        x = x.cpu().detach().numpy()
        predictions = fun(*[x[:, i] > 0.5 for i in range(x.shape[1])])

        # get accuracy
        if expanded_y is not None:
            expanded_y = expanded_y[:, target_class]
            accuracy = f1_score(expanded_y, predictions, average='macro')
        else:
            accuracy = f1_score(y, predictions, average='macro')
        return accuracy, predictions


def _simplify_formula(explanation: str, x: torch.Tensor, y: torch.Tensor, target_class: int, max_accuracy: bool, expanded_y=None) -> str:
    """
    Simplify formula to a simpler one that is still coherent.

    :param explanation: local formula to be simplified.
    :param x: input data.
    :param y: target labels (1D, categorical NOT one-hot encoded).
    :param target_class: target class
    :param max_accuracy: drop  term only if it gets max accuracy
    :return: Simplified formula
    """

    base_accuracy, _ = test_explanation(explanation, x, y, target_class, expanded_y)
    for term in explanation.split(' & '):
        explanation_simplified = copy.deepcopy(explanation)

        if explanation_simplified.endswith(f'{term}'):
            explanation_simplified = explanation_simplified.replace(f' & {term}', '')
        else:
            explanation_simplified = explanation_simplified.replace(f'{term} & ', '')

        if explanation_simplified:
            accuracy, preds = test_explanation(explanation_simplified, x, y, target_class, expanded_y)
            if (max_accuracy and accuracy == 1.) or (not max_accuracy and accuracy >= base_accuracy):
                explanation = copy.deepcopy(explanation_simplified)
                base_accuracy = accuracy

    return explanation


def _aggregate_explanations(local_explanations_accuracy, topk_explanations, target_class, x, y, expanded_y=None):
    """
    Sort explanations by accuracy and then aggregate explanations which increase the accuracy of the aggregated formula.

    :param local_explanations_accuracy: dictionary of explanations and related accuracies.
    :param topk_explanations: limits the number of explanations to be aggregated.
    :param target_class: target class.
    :param x: observations in validation set.
    :param y: labels in validation set.
    :return:
    """
    if len(local_explanations_accuracy) == 0:
        return ''

    else:
        # get the topk most accurate local explanations
        local_explanations_sorted = sorted(local_explanations_accuracy.items(), key=lambda x: -x[1][1])[:topk_explanations]
        explanations = []
        best_accuracy = 0
        best_explanation = ''
        for explanation_raw, (explanation, accuracy) in local_explanations_sorted:
            explanations.append(explanation)

            # aggregate example-level explanations
            aggregated_explanation = ' | '.join(explanations)
            aggregated_explanation_simplified = simplify_logic(aggregated_explanation, 'dnf', force=True)
            aggregated_explanation_simplified = f'({aggregated_explanation_simplified})'

            if aggregated_explanation_simplified in ['', 'False', 'True', '(False)', '(True)']:
                continue
            accuracy, _ = test_explanation(aggregated_explanation_simplified, x, y, target_class, expanded_y)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_explanation = aggregated_explanation_simplified
                explanations = [best_explanation]

    return best_explanation, best_accuracy


# def _local_explanation(module, feature_names, neuron_id, neuron_explanations_raw,
#                        c_validation, y_target, target_class, max_accuracy, max_minterm_complexity):
#     # explanation is the conjunction of non-pruned features
#     explanation_raw = ''
#     if max_minterm_complexity:
#         concepts_to_retain = torch.argsort(module.alpha[target_class], descending=True)[:max_minterm_complexity]
#     else:
#         non_pruned_concepts = module.concept_mask[target_class]
#         concepts_sorted = torch.argsort(module.alpha[target_class])
#         concepts_to_retain = concepts_sorted[non_pruned_concepts[concepts_sorted]]
#
#     for j in concepts_to_retain:
#         sub_feature_names = [f'{feature_names[j]}_{k:010}' for k in range(c_validation.size(-2))]
#         for i in range(c_validation.shape[1]):
#             if explanation_raw:
#                 explanation_raw += ' & '
#             if c_validation[neuron_id, i, j] > 0.5:
#                 # if non_pruned_neurons[j] > 0:
#                 explanation_raw += sub_feature_names[i]
#             else:
#                 explanation_raw += f'~{sub_feature_names[i]}'
#
#     explanation_raw = str(explanation_raw)
#     if explanation_raw in ['', 'False', 'True', '(False)', '(True)']:
#         return None, None
#
#     simplify = True
#     if explanation_raw in neuron_explanations_raw:
#         explanation = neuron_explanations_raw[explanation_raw]
#     elif simplify:
#         explanation = _simplify_formula(explanation_raw, c_validation, y_target, target_class, max_accuracy)
#     else:
#         explanation = explanation_raw
#
#     if explanation in ['', 'False', 'True', '(False)', '(True)']:
#         return None, None
#
#     return explanation, explanation_raw

def _local_explanation(module, feature_names, neuron_id, neuron_explanations_raw,
                       c_validation, y_target, target_class, max_accuracy, max_minterm_complexity, expanded_y_target=None):
    # explanation is the conjunction of non-pruned features
    explanation_raw = ''
    if max_minterm_complexity:
        concepts_to_retain = torch.argsort(module.alpha[target_class], descending=True)[:max_minterm_complexity]
    else:
        non_pruned_concepts = module.concept_mask[target_class]
        concepts_sorted = torch.argsort(module.alpha[target_class])
        concepts_to_retain = concepts_sorted[non_pruned_concepts[concepts_sorted]]

    for j in concepts_to_retain:
        if feature_names[j] not in ['()', '']:
            if explanation_raw:
                explanation_raw += ' & '
            if module.conceptizator.concepts[0][neuron_id, j] > module.conceptizator.threshold:
                # if non_pruned_neurons[j] > 0:
                explanation_raw += feature_names[j]
            else:
                explanation_raw += f'~{feature_names[j]}'

    explanation_raw = str(explanation_raw)
    if explanation_raw in ['', 'False', 'True', '(False)', '(True)']:
        return None, None

    simplify = True
    if explanation_raw in neuron_explanations_raw:
        explanation = neuron_explanations_raw[explanation_raw]
    elif simplify:
        explanation = _simplify_formula(explanation_raw, c_validation, y_target, target_class, max_accuracy, expanded_y_target)
    else:
        explanation = explanation_raw

    if explanation in ['', 'False', 'True', '(False)', '(True)']:
        return None, None

    return explanation, explanation_raw


def _get_correct_data_pooled(x, y, model, target_class, batch, expanded_y):
    x_target = x[expanded_y[:, target_class] == 1]
    y_target = y[y[:, target_class] == 1]

    # get model's predictions
    target_batch = relabel_batch(batch[expanded_y[:, target_class] == 1]).long()
    preds = model(x_target, target_batch).squeeze(-1)

    # identify samples correctly classified of the target class
    correct_mask_y = y_target[:, target_class].eq(preds[:, target_class]>0.5)
    if sum(correct_mask_y) < 2:
        return None, None, None

    # create a mask that fits x input given y input
    correct_mask_x = reshape_graph_to_node_data(correct_mask_y, target_batch)

    x_target_correct = x_target[correct_mask_x]
    y_target_correct = y_target[correct_mask_y]
    batch_target_correct = relabel_batch(target_batch[correct_mask_x])

    # collapse samples having the same boolean values and class label different from the target class
    x_reduced_opposite = x[expanded_y[:, target_class] != 1]
    y_reduced_opposite = y[y[:, target_class] != 1]

    # get model's predictions
    target_batch_opposite = relabel_batch(batch[expanded_y[:, target_class] != 1]).long()
    preds_opposite = model(x_reduced_opposite, target_batch_opposite).squeeze(-1)

    # identify samples correctly classified of the opposite class
    correct_mask_y = y_reduced_opposite[:, target_class].eq(preds_opposite[:, target_class]>0.5)
    if sum(correct_mask_y) < 2:
        return None, None, None

    # create a mask that fits x input given y input
    correct_mask_x = reshape_graph_to_node_data(correct_mask_y, target_batch_opposite)

    x_reduced_opposite_correct = x_reduced_opposite[correct_mask_x]
    y_reduced_opposite_correct = y_reduced_opposite[correct_mask_y]
    batch_reduced_opposite_correct = relabel_batch(target_batch_opposite[correct_mask_x])

    # select the subset of samples belonging to the target class
    x_validation = torch.cat([x_reduced_opposite_correct, x_target_correct], dim=0)
    y_validation = torch.cat([y_reduced_opposite_correct, y_target_correct], dim=0)

    offset = batch_reduced_opposite_correct[-1] + 1
    batch_validation = torch.cat([batch_reduced_opposite_correct, batch_target_correct + offset])

    expanded_y_validation = reshape_graph_to_node_data(y_validation, batch_validation)

    # model.eval()
    # model(x_validation, batch_validation)

    return x_validation, y_validation, expanded_y_validation


def _get_correct_data(x, y, model, target_class):
    x_target = x[y[:, target_class] == 1]
    y_target = y[y[:, target_class] == 1]

    # get model's predictions
    preds = model(x_target).squeeze(-1)

    # identify samples correctly classified of the target class
    correct_mask = y_target[:, target_class].eq(preds[:, target_class]>0.5)
    if sum(correct_mask) < 2:
        return None, None

    x_target_correct = x_target[correct_mask]
    y_target_correct = y_target[correct_mask]

    # collapse samples having the same boolean values and class label different from the target class
    x_reduced_opposite = x[y[:, target_class] != 1]
    y_reduced_opposite = y[y[:, target_class] != 1]

    # get model's predictions
    preds_opposite = model(x_reduced_opposite).squeeze(-1)


    # identify samples correctly classified of the opposite class
    correct_mask = y_reduced_opposite[:, target_class].eq(preds_opposite[:, target_class]>0.5)
    if sum(correct_mask) < 2:
        return None, None

    x_reduced_opposite_correct = x_reduced_opposite[correct_mask]
    y_reduced_opposite_correct = y_reduced_opposite[correct_mask]

    # select the subset of samples belonging to the target class
    x_validation = torch.cat([x_reduced_opposite_correct, x_target_correct], dim=0)
    y_validation = torch.cat([y_reduced_opposite_correct, y_target_correct], dim=0)

    model.eval()
    # get model's predictions
    model(x_validation)

    return x_validation, y_validation


def aggregate_expalanations(explanations_data_list):
    running_explanation_accuracy = 0.0
    running_explanations_complexity = 0.0

    for explanations_data in enumerate(explanations_data_list):
        print(explanation)
        running_explanation_accuracy += explanations_accuracy['expl_acc']
        running_explanations_complexity += explanations_accuracy['expl_complex']

    divisor = len(explanations_data_list)
    print(f"Average Explanation Accuracy: {running_explanation_accuracy / divisor:.4f}")
    print(f"Average Explanation Complexity: {running_explanation_complexity / divisor:.4f}")
