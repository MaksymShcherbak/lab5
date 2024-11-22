# create_tables.py
from database import engine
import models

# Create all tables defined in the models (including User)
models.Base.metadata.create_all(bind=engine)
