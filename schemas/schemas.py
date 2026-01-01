from datetime import datetime
from uuid import UUID
from typing import Optional

from pydantic import BaseModel, Field


# ============== Conversation Schemas ==============

class ConversationCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None


class ConversationResponse(BaseModel):
    id: UUID
    name: str
    description: Optional[str]
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ============== Message Schemas ==============

class MessageResponse(BaseModel):
    id: UUID
    role: str
    content: str
    created_at: datetime

    model_config = {"from_attributes": True, "use_enum_values": True}


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    message: MessageResponse
    conversation_id: UUID


# ============== Image Schemas ==============

class ImageResponse(BaseModel):
    id: UUID
    conversation_id: UUID
    filename: str
    storage_url: str
    content_type: Optional[str]
    file_size: Optional[int]
    status: str
    created_at: datetime

    model_config = {"from_attributes": True, "use_enum_values": True}


class UploadResponse(BaseModel):
    message: str
    image: ImageResponse


class ConversationWithImagesResponse(BaseModel):
    id: UUID
    name: str
    description: Optional[str]
    created_at: datetime
    updated_at: datetime
    images: list[ImageResponse]

    model_config = {"from_attributes": True}


class ConversationDetailResponse(BaseModel):
    id: UUID
    name: str
    description: Optional[str]
    created_at: datetime
    updated_at: datetime
    images: list[ImageResponse]
    messages: list[MessageResponse]

    model_config = {"from_attributes": True}


# ============== Detection Schemas ==============

class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float


class DetectedObjectResponse(BaseModel):
    id: UUID
    label: str
    confidence: float
    bbox: Optional[BoundingBox] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class DetectRequest(BaseModel):
    image_id: UUID


class DetectResponse(BaseModel):
    image_id: UUID
    objects: list[DetectedObjectResponse]


# ============== OCR Schemas ==============

class ExtractedTextResponse(BaseModel):
    id: UUID
    text: str
    confidence: Optional[float]
    position_x: Optional[float]
    position_y: Optional[float]
    created_at: datetime

    model_config = {"from_attributes": True}


class OCRRequest(BaseModel):
    image_id: UUID


class OCRResponse(BaseModel):
    image_id: UUID
    texts: list[ExtractedTextResponse]


# ============== Full Analysis Schemas ==============

class AnalyzeRequest(BaseModel):
    conversation_id: UUID
    context: Optional[str] = Field(None, description="Additional context or description about the case")
    model: str = Field("local", pattern="^(local|gpt4o)$", description="Model to use: 'local' (YOLO+OCR) or 'gpt4o'")


class ImageAnalysisResult(BaseModel):
    image_id: UUID
    detected_objects: list[DetectedObjectResponse]
    extracted_texts: list[ExtractedTextResponse]


class AnalysisResult(BaseModel):
    status: str
    images: list[ImageAnalysisResult]
    hypothesis: str


# ============== Error Schemas ==============

class ErrorResponse(BaseModel):
    detail: str
