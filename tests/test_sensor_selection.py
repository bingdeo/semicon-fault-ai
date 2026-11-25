import numpy as np
import pandas as pd

from src.features import SensorSelector


def test_sensor_selection_keeps_fitted_columns_and_raw_missing_values() -> None:
    train = pd.DataFrame({
        "signal": [0.0, 1.0, 0.0, 1.0, 7.0, 8.0, 7.0, 8.0],
        "noise": [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0],
    })
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    validation = pd.DataFrame({"noise": [100.0, -100.0], "signal": [np.nan, 3.0]})

    selector = SensorSelector(k=1).fit(train, labels)

    assert selector.selected_columns_ == ["signal"]
    pd.testing.assert_frame_equal(selector.transform(validation), validation[["signal"]])
