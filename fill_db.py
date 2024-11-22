import random
from faker import Faker
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base, User, Item, Rating

# Set up Faker and SQLAlchemy session
fake = Faker()
DATABASE_URL = "sqlite:///./test.db"  # Change this if you're using another database
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create the database tables
Base.metadata.create_all(bind=engine)


# Function to generate dummy users
def create_dummy_users(db, num_users=10):
    users = []
    for _ in range(num_users):
        user = User(
            username=fake.user_name(),
            email=fake.email(),
            password=fake.password(),
            preferences=fake.text(max_nb_chars=50),
        )
        db.add(user)
        users.append(user)
    db.commit()
    return users


# Function to generate dummy items
def create_dummy_items(db, num_items=20):
    items = []
    for _ in range(num_items):
        item = Item(
            name=fake.word().capitalize() + " " + fake.word().capitalize(),
            category=fake.word().capitalize(),
            description=fake.text(max_nb_chars=200),
            price=round(random.uniform(5.0, 100.0), 2),
            stock_quantity=random.randint(0, 50),
        )
        db.add(item)
        items.append(item)
    db.commit()
    return items


# Function to generate random ratings for users and items
def create_dummy_ratings(db, users, items, num_ratings=50):
    for _ in range(num_ratings):
        user = random.choice(users)
        item = random.choice(items)
        rating = Rating(
            rating=random.randint(1, 5),  # Rating between 1 and 5
            product_id=item.id,
            user_id=user.id,
        )
        db.add(rating)
    db.commit()


# Main function to populate the database with dummy data
def populate_db():
    # Create a new session
    db = SessionLocal()

    try:
        # Create dummy users, items, and ratings
        users = create_dummy_users(db, num_users=10)  # You can change the number
        items = create_dummy_items(db, num_items=20)  # You can change the number
        create_dummy_ratings(
            db, users, items, num_ratings=50
        )  # You can change the number

        print("Database populated with dummy data.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    populate_db()
