# data_sources.py - one place that knows about the TWO data sources
#   "mock"  -> retail_data.db  (75 days of generated data)
#   "video" -> video_data.db   (written by vision_pipeline.py from a real video)
# Both databases use the SAME table layout defined in database.py (which is NOT modified).
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database import Base

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SOURCES = {
    "mock": os.path.join(BASE_DIR, "retail_data.db"),
    "video": os.path.join(BASE_DIR, "video_data.db"),
}

_engines = {}


def get_engine(source):
    if source not in SOURCES:
        raise ValueError(f"Unknown data source '{source}'. Use one of {list(SOURCES)}")
    if source not in _engines:
        eng = create_engine(f"sqlite:///{SOURCES[source]}",
                            connect_args={"check_same_thread": False})
        Base.metadata.create_all(eng)      # creates the table if the file is new
        _engines[source] = eng
    return _engines[source]


def get_session(source):
    return sessionmaker(bind=get_engine(source))()