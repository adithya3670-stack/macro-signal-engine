# Third-Party Notices

This project includes open-source dependencies distributed under their
respective licenses.

The list below is a curated core set of major runtime/training/test libraries.
For full transitive dependency resolution, see:

- `pyproject.toml`
- `requirements/runtime.lock.txt`
- `requirements/train.lock.txt`
- `requirements/research.lock.txt`
- `requirements/test.lock.txt`

License references are based on upstream project declarations and should be
re-verified before redistribution in regulated/commercial environments.

## Core Dependencies

| Package | Primary Use | License | Upstream |
| --- | --- | --- | --- |
| Flask | Web framework | BSD-3-Clause | https://github.com/pallets/flask |
| pandas | Data processing | BSD-3-Clause | https://github.com/pandas-dev/pandas |
| NumPy | Numerical computing | BSD-3-Clause | https://github.com/numpy/numpy |
| requests | HTTP client | Apache-2.0 | https://github.com/psf/requests |
| python-dotenv | Environment loading | BSD-3-Clause | https://github.com/theskumar/python-dotenv |
| joblib | Serialization/utilities | BSD-3-Clause | https://github.com/joblib/joblib |
| scikit-learn | ML utilities/metrics | BSD-3-Clause | https://github.com/scikit-learn/scikit-learn |
| Plotly | Visualization | MIT | https://github.com/plotly/plotly.py |
| openpyxl | Excel export support | MIT | https://foss.heptapod.net/openpyxl/openpyxl |
| gunicorn | WSGI server | MIT | https://github.com/benoitc/gunicorn |
| PyTorch (`torch`) | Deep learning runtime | BSD-3-Clause | https://github.com/pytorch/pytorch |
| XGBoost | Gradient boosting training | Apache-2.0 | https://github.com/dmlc/xgboost |
| matplotlib | Research plotting | Matplotlib License (BSD-style) | https://github.com/matplotlib/matplotlib |
| seaborn | Statistical visualization | BSD-3-Clause | https://github.com/mwaskom/seaborn |
| pytest | Test framework | MIT | https://github.com/pytest-dev/pytest |
| pytest-cov | Test coverage plugin | MIT | https://github.com/pytest-dev/pytest-cov |

## Notes

- This notices file does not replace individual upstream license texts.
- Consumers are responsible for validating license obligations for their usage model.
