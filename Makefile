# ============================================================
# Makefile — Behavioral Scoring MLOps
# ============================================================

install:
	pip install -r requirements.txt

# Primera ejecucion: corre GridSearchCV y genera los pkl
train:
	python train_pipeline.py reentrenar=true

# Ejecuciones siguientes: carga los pkl sin reentrenar
run:
	python train_pipeline.py

# Prediccion sobre conjunto de prueba
predict:
	python predict.py

# Limpiar outputs (no borra el CSV original)
clean:
	rm -f models/*.pkl
	rm -f reports/*.png reports/*.csv

# Ver configuracion activa
config:
	python -c "import hydra; print('Hydra OK')"

# Scoring de cartera vigente (genera reports/scoring/)
score:
	python score_cartera.py

# Dashboard Streamlit
dashboard:
	python -m streamlit run app/Inicio.py

# Congelar transformador (una vez; train_pipeline lo repite al reentrenar)
congelar:
	python congelar_transformador.py

# Incorporar un corte mensual: make corte ARCHIVO=ruta.csv
corte:
	python actualizar_corte.py $(ARCHIVO)
