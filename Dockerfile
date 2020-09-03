FROM bempp/notebook
RUN pip install pandas cython>=0.23
COPY bempp_cavity bempp_cavity
COPY setup.py setup.py
COPY README.md README.md
RUN pip install -e .
