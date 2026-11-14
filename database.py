import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "retail_data.db")

engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
Session = sessionmaker(bind=engine)
Base = declarative_base()

class VisitorEvent(Base):
    __tablename__ = "visitor_events"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    track_id = Column(Integer)
    gender = Column(String)
    age_group = Column(String)
    zone = Column(String)
    dwell_seconds = Column(Float)
    timestamp = Column(DateTime)