# models.py
from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Item(Base):
    __tablename__ = "items"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    category = Column(String)  # Example: "Rulebook", "Miniature", etc.
    description = Column(String)
    price = Column(Float)
    stock_quantity = Column(Integer)
    ratings = relationship("Rating", back_populates="product")


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    password = Column(String)
    preferences = Column(String, default="")
    ratings = relationship("Rating", back_populates="user")


class Rating(Base):
    __tablename__ = "ratings"

    id = Column(Integer, primary_key=True, index=True)
    rating = Column(Integer)  # Assuming the rating is an integer, but can be customized
    product_id = Column(
        Integer, ForeignKey("items.id")
    )  # Assuming "items" table exists
    user_id = Column(Integer, ForeignKey("users.id"))  # Assuming "users" table exists

    # Relationships to the corresponding tables
    product = relationship("Item", back_populates="ratings")
    user = relationship("User", back_populates="ratings")
