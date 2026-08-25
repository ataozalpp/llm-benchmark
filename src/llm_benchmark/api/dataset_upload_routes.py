"""Dataset upload API routes."""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Request,
    UploadFile,
    status,
)
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from llm_benchmark.dataset_ingestion import (
    DatasetIngestionRequest,
    DatasetIngestionService,
)

from .app_dependencies import (
    get_dataset_ingestion_service,
)
from .schemas import (
    DatasetResponse,
    DatasetUploadForm,
)

router = APIRouter(
    prefix="/datasets",
    tags=["datasets"],
)


_UPLOAD_FORM_FIELDS = {
    "file",
    "name",
    "file_format",
    "split",
    "revision",
    "license",
}


async def _dataset_upload_form(
    request: Request,
    name: Annotated[str, Form(...)],
    file_format: Annotated[Literal["csv", "jsonl"], Form(...)],
    split: Annotated[str, Form()] = "test",
    revision: Annotated[str | None, Form()] = None,
    license: Annotated[str | None, Form()] = None,
) -> DatasetUploadForm:
    submitted = {
        "name": name,
        "file_format": file_format,
        "split": split,
        "revision": revision,
        "license": license,
    }
    multipart = await request.form()
    for field_name in multipart:
        if field_name not in _UPLOAD_FORM_FIELDS:
            submitted[field_name] = None

    try:
        return DatasetUploadForm.model_validate(submitted)
    except ValidationError as error:
        safe_errors = [
            {
                "loc": ("body", *item["loc"]),
                "msg": item["msg"],
                "type": item["type"],
            }
            for item in error.errors()
        ]
        raise RequestValidationError(safe_errors) from error


@router.post(
    "/upload",
    response_model=DatasetResponse,
    status_code=status.HTTP_201_CREATED,
)
def upload_dataset(
    file: Annotated[UploadFile, File(...)],
    form: Annotated[DatasetUploadForm, Depends(_dataset_upload_form)],
    service: Annotated[
        DatasetIngestionService,
        Depends(get_dataset_ingestion_service),
    ],
) -> DatasetResponse:
    result = service.ingest(
        DatasetIngestionRequest(
            name=form.name,
            file_format=form.file_format,
            split=form.split,
            revision=form.revision,
            license=form.license,
        ),
        file.file,
    )
    return DatasetResponse.from_record(result.dataset)
