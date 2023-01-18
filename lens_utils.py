import torch_explain as te
from torch_explain.logic.nn import entropy
from torch_explain.logic.metrics import test_explanation, complexity
import torch.nn.functional as F

def explain_classes(model, concepts, y, train_mask, test_mask, max_minterm_complexity=1000, topk_explanations=1000, try_all=False):
    y1h = F.one_hot(y[train_mask])
    y1h_test = F.one_hot(y[test_mask])

    explanations = {}

    for class_id in range(y1h.shape[1]):
        explanation, _ = entropy.explain_class(model.lens, concepts[train_mask], y1h,
                                               concepts[train_mask], y1h, target_class=class_id,
                                               max_minterm_complexity=max_minterm_complexity,
                                               topk_explanations=topk_explanations, try_all=try_all)



        explanation_accuracy, _ = test_explanation(explanation, concepts[test_mask], y1h_test, target_class=class_id)
        explanation_complexity = complexity(explanation)

        explanations[str(class_id)] = {'explanation': explanation,
                                        'explanation_accuracy': explanation_accuracy,
                                        'explanation_complexity': explanation_complexity}

        print(f'Explanation class {class_id}: {explanation} - acc. = {explanation_accuracy:.4f} - compl. = {explanation_complexity:.4f}')

    return explanations


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
