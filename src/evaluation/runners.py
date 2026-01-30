from src.evaluation.classification import compute_classification_accuracy
from src.evaluation.ed_cost import compute_ed_cost

def run_standard_evaluation(model, ed_layer, data_loader, device):
    model.eval()

    results = {}

    # Get classification accuracy
    classification_accuracy = compute_classification_accuracy(model, data_loader, device)

    # Get ED cost
    ed_cost_results = compute_ed_cost(model, ed_layer, data_loader, device)

    # Get competition score


    return None