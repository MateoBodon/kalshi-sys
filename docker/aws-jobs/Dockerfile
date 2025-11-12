FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml poetry.lock* requirements*.txt* /app/
RUN pip install --no-cache-dir "polars==0.20.31" "pyarrow==17.0.0"
COPY . /app
RUN pip install --no-cache-dir -e .
ENTRYPOINT ["python"]
