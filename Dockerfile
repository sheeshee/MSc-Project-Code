FROM bempp/base
RUN pip install --upgrade pip --user
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --user
