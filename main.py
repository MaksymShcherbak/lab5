from fastapi import FastAPI, Depends, HTTPException, Form, Request, Response, status
from fastapi import Cookie
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session
from passlib.context import CryptContext
import jwt
from datetime import datetime, timedelta
import models
import database

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Or use "*" for all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Password hashing setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Secret key for JWT encoding/decoding
SECRET_KEY = "mysecretkey"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


# Dependency to get the database session
def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Utility function to verify the password
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


# Utility function to hash passwords
def get_password_hash(password):
    return pwd_context.hash(password)


# Create an access token (JWT)
def create_access_token(
    data: dict,
    expires_delta: timedelta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


# Routes for managing items


@app.post("/items/", response_model=dict)
def create_item(
    name: str = Form(...),
    category: str = Form(...),
    description: str = Form(...),
    price: float = Form(...),
    stock_quantity: int = Form(...),
    db: Session = Depends(get_db),
):
    db_item = models.Item(
        name=name,
        category=category,
        description=description,
        price=price,
        stock_quantity=stock_quantity,
    )
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return {
        "message": f"Item {db_item.name} created successfully!",
        "item": db_item.name,
    }


@app.get("/items/{item_id}", response_model=dict)
def read_item(item_id: int, db: Session = Depends(get_db)):
    item = db.query(models.Item).filter(models.Item.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return {
        "item_id": item.id,
        "name": item.name,
        "category": item.category,
        "price": item.price,
    }


@app.put("/items/{item_id}", response_model=dict)
def update_item(item_id: int, item: dict, db: Session = Depends(get_db)):
    db_item = db.query(models.Item).filter(models.Item.id == item_id).first()
    if not db_item:
        raise HTTPException(status_code=404, detail="Item not found")

    db_item.name = item["name"]
    db_item.category = item["category"]
    db_item.description = item["description"]
    db_item.price = item["price"]
    db_item.stock_quantity = item["stock_quantity"]

    db.commit()
    db.refresh(db_item)
    return {
        "message": f"Item {db_item.name} updated successfully!",
        "item": db_item.name,
    }


@app.delete("/items/{item_id}", response_model=dict)
def delete_item(item_id: int, db: Session = Depends(get_db)):
    db_item = db.query(models.Item).filter(models.Item.id == item_id).first()
    if not db_item:
        raise HTTPException(status_code=404, detail="Item not found")

    db.delete(db_item)
    db.commit()
    return {"message": f"Item {db_item.name} deleted successfully!"}


# Routes for managing users


@app.post("/users/", response_model=dict)
def create_user(user: dict, db: Session = Depends(get_db)):
    db_user = models.User(
        username=user["username"], email=user["email"], preferences=user["preferences"]
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return {
        "message": f"User {db_user.username} created successfully!",
        "user": db_user.username,
    }


@app.get("/users/{user_id}", response_model=dict)
def read_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "user_id": user.id,
        "username": user.username,
        "email": user.email,
        "preferences": user.preferences,
    }


@app.put("/users/{user_id}", response_model=dict)
def update_user(user_id: int, user: dict, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")

    db_user.username = user["username"]
    db_user.email = user["email"]
    db_user.preferences = user["preferences"]

    db.commit()
    db.refresh(db_user)
    return {
        "message": f"User {db_user.username} updated successfully!",
        "user": db_user.username,
    }


@app.delete("/users/{user_id}", response_model=dict)
def delete_user(user_id: int, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")

    db.delete(db_user)
    db.commit()
    return {"message": f"User {db_user.username} deleted successfully!"}


# Admin Routes (for adding/editing items) - Not Protected
@app.get("/admin/items", response_class=HTMLResponse)
async def admin_items(request: Request, db: Session = Depends(get_db)):
    items = db.query(models.Item).all()
    return templates.TemplateResponse(
        "admin_items.html", {"request": request, "items": items}
    )


@app.get("/admin/items/add", response_class=HTMLResponse)
async def add_item_page(request: Request):
    return templates.TemplateResponse("add_item.html", {"request": request})


@app.post("/admin/items/add")
async def add_item(
    name: str = Form(...),
    description: str = Form(...),
    price: float = Form(...),
    db: Session = Depends(get_db),
):
    new_item = models.Item(name=name, description=description, price=price)
    db.add(new_item)
    db.commit()
    db.refresh(new_item)
    return {"message": f"Item '{new_item.name}' added successfully!"}


# User Registration (Sign-Up)
@app.get("/signup", response_class=HTMLResponse)
async def signup_page(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})


@app.post("/signup")
async def signup(
    response: Response,  # response is non-default here, no `Depends()`
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),  # db dependency comes after non-default args
):
    # Check if the email is already taken
    db_user = db.query(models.User).filter(models.User.email == email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_password = get_password_hash(password)
    db_user = models.User(username=username, email=email, password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    # Create JWT token
    access_token = create_access_token(data={"sub": db_user.email})

    response = RedirectResponse(url="/rate_product", status_code=303)

    # Set the JWT token as a cookie in the response
    response.set_cookie(
        key="access_token",
        value=access_token,
        max_age=36000,  # 10 hoursr "Lax" for better security, especially in cross-origin requests
        samesite="None",  # Allow cookies even on redirects
        secure=True,  # Only send cookies over HTTPS (use only for production)
    )

    # Redirect to the rate page
    return response


@app.post("/login")
async def login(
    response: Response,  # response is non-default here, no `Depends()`
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),  # db dependency comes after non-default args
):
    db_user = db.query(models.User).filter(models.User.email == email).first()

    # Password validation (check if user exists and password matches)
    if not db_user or not verify_password(password, db_user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Create JWT token
    access_token = create_access_token(data={"sub": db_user.email})

    response = RedirectResponse(url="/rate_product", status_code=303)

    # Set the JWT token as a cookie in the response
    response.set_cookie(
        key="access_token",
        value=access_token,
        max_age=36000,  # 10 hoursr "Lax" for better security, especially in cross-origin requests
        samesite="None",  # Allow cookies even on redirects
        secure=True,  # Only send cookies over HTTPS (use only for production)
    )

    # Redirect to the rate page
    return response


# User Login
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


# main.py
@app.post("/rate")
async def rate_product(
    product_id: int = Form(...),
    rating: int = Form(...),
    access_token: str = Cookie(None),  # Get the token from the cookie
    db: Session = Depends(get_db),
):
    if not access_token:
        raise HTTPException(status_code=401, detail="Token is missing or expired")

    try:
        # Decode the JWT token
        payload = jwt.decode(access_token, SECRET_KEY, algorithms=[ALGORITHM])
        user_email = payload.get("sub")
        if user_email is None:
            raise HTTPException(status_code=401, detail="Invalid token")

        db_user = db.query(models.User).filter(models.User.email == user_email).first()
        if not db_user:
            raise HTTPException(status_code=404, detail="User not found")

        # Check if the product exists
        db_product = db.query(models.Item).filter(models.Item.id == product_id).first()
        if not db_product:
            raise HTTPException(status_code=404, detail="Product not found")

        # Check if the user has already rated this product
        existing_rating = (
            db.query(models.Rating)
            .filter(
                models.Rating.product_id == product_id,
                models.Rating.user_id == db_user.id,
            )
            .first()
        )

        if existing_rating:
            # If the user has already rated the product, update the existing rating
            existing_rating.rating = rating
            db.commit()
            db.refresh(existing_rating)
            return {
                "message": f"User {db_user.username} updated their rating for product {product_id} to {rating}"
            }
        else:
            # If the user hasn't rated the product, create a new rating
            new_rating = models.Rating(
                rating=rating, product_id=product_id, user_id=db_user.id
            )
            db.add(new_rating)
            db.commit()
            db.refresh(new_rating)

        return RedirectResponse(url="/rate_product", status_code=303)
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


@app.post("/logout")
async def logout(response: Response):
    response.delete_cookie("access_token")
    return {"message": "Logged out successfully"}


import numpy as np


# Function to calculate RMSE
def calculate_rmse(predictions, actual_ratings):
    mse = np.mean(
        [(pred - actual) ** 2 for pred, actual in zip(predictions, actual_ratings)]
    )
    return np.sqrt(mse)


# Generate recommendations (just a placeholder here, replace with your actual recommendation logic)
def generate_recommendations(user_id, db):
    # Placeholder logic for recommendations
    return db.query(models.Item).limit(5).all()


@app.get("/rate_product", response_class=HTMLResponse)
async def rate_product_page(
    request: Request, db: Session = Depends(get_db), access_token: str = Cookie(None)
):
    if not access_token:
        raise HTTPException(status_code=401, detail="Token is missing or expired")

    try:
        # Decode the JWT token to get the user email
        payload = jwt.decode(access_token, SECRET_KEY, algorithms=[ALGORITHM])
        user_email = payload.get("sub")
        if user_email is None:
            raise HTTPException(status_code=401, detail="Invalid token")

        # Fetch the user from the database
        db_user = db.query(models.User).filter(models.User.email == user_email).first()
        if not db_user:
            raise HTTPException(status_code=404, detail="User not found")

        # Generate recommendations for the logged-in user
        recommended_items = generate_recommendations(db_user.id, db)

        # Fetch all items and their average ratings
        items = db.query(models.Item).all()

        actual_ratings = []
        predicted_ratings = []

        # Calculate actual and predicted ratings
        for item in items:
            # Average actual rating for the item
            ratings_for_item = [rating.rating for rating in item.ratings]
            avg_actual_rating = (
                sum(ratings_for_item) / len(ratings_for_item) if ratings_for_item else 0
            )
            actual_ratings.append(avg_actual_rating)

            # Predicted rating for the item based on the recommendation system
            predicted_rating = predict_rating(db_user.id, item.id, db)
            predicted_ratings.append(predicted_rating)

        # Calculate RMSE (Root Mean Square Error)
        def calculate_rmse(predicted, actual):
            return (
                sum((p - a) ** 2 for p, a in zip(predicted, actual)) / len(actual)
            ) ** 0.5

        rmse = calculate_rmse(predicted_ratings, actual_ratings)

        # Create a plot (e.g., bar chart of actual vs. predicted ratings)
        import matplotlib.pyplot as plt
        import io
        import base64

        # Generate the plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Use item IDs for x-axis labels
        item_ids = [item.id for item in items]
        bar_width = 0.4  # Width of the bars

        # Create the bars for actual ratings
        actual_bar_positions = np.arange(len(item_ids))
        ax.bar(
            actual_bar_positions,
            actual_ratings,
            width=bar_width,
            label="Actual Ratings",
            color="b",
            alpha=0.6,
        )

        # Create the bars for predicted ratings with slight offset
        predicted_bar_positions = actual_bar_positions + bar_width
        ax.bar(
            predicted_bar_positions,
            predicted_ratings,
            width=bar_width,
            label="Predicted Ratings",
            color="r",
            alpha=0.6,
        )

        # Adjust the x-axis
        ax.set_xticks(actual_bar_positions + bar_width / 2)  # Center the ticks
        ax.set_xticklabels(item_ids)  # Use item IDs as labels
        ax.set_xlabel("Item IDs")
        ax.set_ylabel("Ratings")
        ax.set_title("Actual vs Predicted Ratings")
        ax.legend()

        # Save the plot to a base64 string for embedding in HTML
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        plot_url = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()

        # Return the page with recommendations, RMSE, and the plot
        return templates.TemplateResponse(
            "rate_product.html",
            {
                "request": request,
                "recommended_items": recommended_items,
                "items": items,
                "rmse": rmse,
                "plot_url": plot_url,
            },
        )
    except jwt.PyJWTError as e:
        raise HTTPException(
            status_code=401, detail="Invalid or expired token: " + str(e)
        )


def predict_rating(user_id: int, item_id: int, db: Session) -> float:
    # Fetch all ratings for the item
    item_ratings = (
        db.query(models.Rating).filter(models.Rating.product_id == item_id).all()
    )

    if not item_ratings:
        return 3.0  # Default rating if no ratings exist

    # Collaborative filtering (simple mean of other ratings for this item)
    return sum(rating.rating for rating in item_ratings) / len(item_ratings)


from surprise import Dataset, Reader, SVD
from surprise import accuracy
import pandas as pd


def generate_recommendations(user_id: int, db: Session):
    # Query the ratings from the database
    ratings_data = db.query(models.Rating).all()

    # Create a DataFrame with the ratings data
    ratings_df = pd.DataFrame(
        [(rating.user_id, rating.product_id, rating.rating) for rating in ratings_data],
        columns=["user_id", "product_id", "rating"],
    )

    # Define the format for surprise
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_df[["user_id", "product_id", "rating"]], reader)
    trainset = data.build_full_trainset()

    # Train a recommendation model using SVD (Singular Value Decomposition)
    model = SVD()
    model.fit(trainset)

    # Get top 5 recommendations for the given user
    user_ratings = ratings_df[ratings_df["user_id"] == user_id]
    rated_items = set(user_ratings["product_id"])

    # Get all items and predict ratings for each one
    all_items = db.query(models.Item).all()
    predictions = []
    for item in all_items:
        if item.id not in rated_items:
            pred = model.predict(user_id, item.id)
            predictions.append((item, pred.est))

    # Sort the predictions by estimated rating (highest first)
    predictions.sort(key=lambda x: x[1], reverse=True)

    # Return the top 5 recommendations
    top_recommendations = predictions[:5]
    return [(rec[0], rec[1]) for rec in top_recommendations]
