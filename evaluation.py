# evaluation.py

def evaluate_model(model, pyg_graph):
    # Evaluation function for test accuracy
    model.eval()
    with torch.no_grad():
        out = model(pyg_graph.x, pyg_graph.edge_index)

    preds = out.argmax(dim=1)  # Get the predicted classes
    correct = (preds[pyg_graph.test_mask] == pyg_graph.y[pyg_graph.test_mask]).sum().item()
    accuracy = correct / pyg_graph.test_mask.sum().item()
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy
