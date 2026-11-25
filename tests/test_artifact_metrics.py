def test_accuracy_recomputed_from_predictions():
    import pandas as pd
    df = pd.DataFrame({"y_true": [1,2,3,4], "y_pred": [1,2,3,3]})
    assert (df.y_true == df.y_pred).mean() == 0.75
