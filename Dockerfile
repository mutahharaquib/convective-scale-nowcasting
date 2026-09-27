FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# train on first build if models are missing (quick settings; run train.py locally for full)
RUN test -f models/unet.pt || (N_TRAIN=80 STEPS=600 python train.py && N_TEST=20 python -m eval.run_eval)
EXPOSE 8000
CMD ["uvicorn", "services.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
