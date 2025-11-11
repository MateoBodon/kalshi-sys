FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml poetry.lock* requirements*.txt* /app/
RUN pip install --no-cache-dir polars
COPY . /app
RUN pip install --no-cache-dir -e .
ENTRYPOINT ["python"]
