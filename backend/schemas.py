from pydantic import BaseModel


class OrderFeatures(BaseModel):
    userid: str | None = None
    product_category: str
    product_price: float
    order_quantity: int
    user_age: int
    user_gender: str
    payment_method: str
    shipping_method: str
    discount_applied: float


class PredictionResponse(BaseModel):
    prediction_label: str
    probability: float | None = None
    decision_threshold: float | None = None
    risk_level: str | None = None
    recommendation: str | None = None


class RegisterRequest(BaseModel):
    name: str
    userid: str
    password: str


class LoginRequest(BaseModel):
    userid: str
    password: str


class LoginResponse(BaseModel):
    success: bool
    name: str | None = None


class PredictionHistoryItem(BaseModel):
    id: int
    userid: str | None = None
    product_category: str
    product_price: float
    order_quantity: int
    user_age: int
    user_gender: str
    payment_method: str
    shipping_method: str
    discount_applied: float
    prediction: int
    probability: float
    created_at: str | None = None
