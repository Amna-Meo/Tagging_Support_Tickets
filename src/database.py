import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Text,
    create_engine,
    Index,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

Base = declarative_base()


class Correction(Base):
    __tablename__ = "corrections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticket_id = Column(String(100), nullable=True)
    ticket_text = Column(Text, nullable=False)
    predicted_tags = Column(Text, nullable=False)
    predicted_confidences = Column(Text, nullable=False)
    accepted_tags = Column(Text, nullable=False)
    confidence_delta = Column(Float, nullable=True)
    mode = Column(String(50), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    reviewer_id = Column(String(100), nullable=True)

    __table_args__ = (
        Index("idx_timestamp", "timestamp"),
        Index("idx_mode", "mode"),
    )


class Database:
    def __init__(self, db_path: str = "corrections.db"):
        self.db_path = Path(db_path)
        self.engine = create_engine(f"sqlite:///{self.db_path}")
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def get_session(self) -> Session:
        return self.SessionLocal()

    def add_correction(
        self,
        ticket_text: str,
        predicted_tags: list[str],
        predicted_confidences: list[float],
        accepted_tags: list[str],
        mode: str,
        ticket_id: Optional[str] = None,
        reviewer_id: Optional[str] = None,
    ) -> Correction:
        session = self.get_session()
        try:
            avg_predicted = sum(predicted_confidences) / len(predicted_confidences)
            avg_accepted = 1.0 / len(accepted_tags) if accepted_tags else 0
            confidence_delta = avg_accepted - avg_predicted

            correction = Correction(
                ticket_id=ticket_id,
                ticket_text=ticket_text,
                predicted_tags=json.dumps(predicted_tags),
                predicted_confidences=json.dumps(predicted_confidences),
                accepted_tags=json.dumps(accepted_tags),
                confidence_delta=confidence_delta,
                mode=mode,
                reviewer_id=reviewer_id,
            )
            session.add(correction)
            session.commit()
            session.refresh(correction)
            return correction
        finally:
            session.close()

    def get_corrections(self, limit: int = 100, offset: int = 0) -> list[Correction]:
        session = self.get_session()
        try:
            return (
                session.query(Correction)
                .order_by(Correction.timestamp.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )
        finally:
            session.close()

    def get_correction_count(self) -> int:
        session = self.get_session()
        try:
            return session.query(Correction).count()
        finally:
            session.close()

    def export_all(self) -> list[dict]:
        session = self.get_session()
        try:
            corrections = session.query(Correction).all()
            return [
                {
                    "text": c.ticket_text,
                    "label": json.loads(c.accepted_tags)[0]
                    if json.loads(c.accepted_tags)
                    else None,
                    "ticket_id": c.ticket_id,
                    "timestamp": c.timestamp.isoformat() if c.timestamp else None,
                }
                for c in corrections
            ]
        finally:
            session.close()


_db_instance: Optional[Database] = None


def init_db(db_path: str = "corrections.db") -> Database:
    global _db_instance
    if _db_instance is None:
        _db_instance = Database(db_path)
    return _db_instance


def get_db() -> Database:
    global _db_instance
    if _db_instance is None:
        return init_db()
    return _db_instance
