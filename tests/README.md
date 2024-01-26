<!-- markdown-toc start - Don't edit this section. Run M-x markdown-toc-refresh-toc -->
# Table of Contents

- [HyVR Tests](#hyvr-tests)
    - [Default test](#default-test)
    - [TODOs](#todos)
- [Coverage](#coverage)

<!-- markdown-toc end -->


# HyVR Tests

In order to ensure that HyVR works even after changing something, several tests
have been set up. Tests are created and run using `pytest`.

## Default test

so far, only simple tests have been created, in the folder tests.
The code has been significantly refactored from the original scritps.

To run the tests, type in the terminal:
```
pytest
```
from the main project directory.
  
# Coverage

Test coverage can be tested using
[coverage](https://coverage.readthedocs.io/en/coverage-5.0.2/). You can run it
from the top level directory via
```
coverage run run_all_tests.py
```
and generate html reports with
```
coverage html
```
The results can be viewed by opening `htmlcov/index.html`.
