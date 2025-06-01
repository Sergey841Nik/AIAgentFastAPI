from sqlalchemy import LargeBinary
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    __abstract__ = True

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)


class User(Base):
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(unique=True, nullable=False)
    password: Mapped[bytes] = mapped_column(LargeBinary)

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.id})"

