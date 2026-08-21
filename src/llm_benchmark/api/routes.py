"""Thin provider-neutral registry routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response, status

from llm_benchmark.db.registry import RegistryRepositories

from .app_dependencies import get_registry
from .run_routes import router as run_router
from .schemas import (
    DatasetCreate,
    DatasetPatch,
    DatasetResponse,
    EndpointCreate,
    EndpointPatch,
    EndpointResponse,
    ModelCreate,
    ModelPatch,
    ModelResponse,
)

router = APIRouter(prefix="/api/v1")

router.include_router(run_router)


@router.post("/endpoints", response_model=EndpointResponse, status_code=status.HTTP_201_CREATED)
def create_endpoint(payload: EndpointCreate, registry: RegistryRepositories = Depends(get_registry)) -> EndpointResponse:
    return EndpointResponse.from_record(registry.endpoints.create(**payload.model_dump()))


@router.get("/endpoints", response_model=list[EndpointResponse])
def list_endpoints(registry: RegistryRepositories = Depends(get_registry)) -> list[EndpointResponse]:
    return [EndpointResponse.from_record(record) for record in registry.endpoints.list_active()]


@router.get("/endpoints/{endpoint_id}", response_model=EndpointResponse)
def get_endpoint(endpoint_id: int, registry: RegistryRepositories = Depends(get_registry)) -> EndpointResponse:
    return EndpointResponse.from_record(registry.endpoints.get_by_id(endpoint_id))


@router.patch("/endpoints/{endpoint_id}", response_model=EndpointResponse)
def update_endpoint(
    endpoint_id: int,
    payload: EndpointPatch,
    registry: RegistryRepositories = Depends(get_registry),
) -> EndpointResponse:
    return EndpointResponse.from_record(
        registry.endpoints.update(endpoint_id, **payload.model_dump(exclude_unset=True))
    )


@router.delete("/endpoints/{endpoint_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_endpoint(endpoint_id: int, registry: RegistryRepositories = Depends(get_registry)) -> Response:
    registry.endpoints.soft_delete(endpoint_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/models", response_model=ModelResponse, status_code=status.HTTP_201_CREATED)
def create_model(payload: ModelCreate, registry: RegistryRepositories = Depends(get_registry)) -> ModelResponse:
    return ModelResponse.from_record(registry.models.create(**payload.model_dump()))


@router.get("/models", response_model=list[ModelResponse])
def list_models(registry: RegistryRepositories = Depends(get_registry)) -> list[ModelResponse]:
    return [ModelResponse.from_record(record) for record in registry.models.list_active()]


@router.get("/models/{model_id}", response_model=ModelResponse)
def get_model(model_id: int, registry: RegistryRepositories = Depends(get_registry)) -> ModelResponse:
    return ModelResponse.from_record(registry.models.get_by_id(model_id))


@router.patch("/models/{model_id}", response_model=ModelResponse)
def update_model(
    model_id: int,
    payload: ModelPatch,
    registry: RegistryRepositories = Depends(get_registry),
) -> ModelResponse:
    return ModelResponse.from_record(registry.models.update(model_id, **payload.model_dump(exclude_unset=True)))


@router.delete("/models/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_model(model_id: int, registry: RegistryRepositories = Depends(get_registry)) -> Response:
    registry.models.soft_delete(model_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/datasets", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
def create_dataset(payload: DatasetCreate, registry: RegistryRepositories = Depends(get_registry)) -> DatasetResponse:
    return DatasetResponse.from_record(registry.datasets.create(**payload.model_dump()))


@router.get("/datasets", response_model=list[DatasetResponse])
def list_datasets(registry: RegistryRepositories = Depends(get_registry)) -> list[DatasetResponse]:
    return [DatasetResponse.from_record(record) for record in registry.datasets.list_active()]


@router.get("/datasets/{dataset_id}", response_model=DatasetResponse)
def get_dataset(dataset_id: int, registry: RegistryRepositories = Depends(get_registry)) -> DatasetResponse:
    return DatasetResponse.from_record(registry.datasets.get_by_id(dataset_id))


@router.patch("/datasets/{dataset_id}", response_model=DatasetResponse)
def update_dataset(
    dataset_id: int,
    payload: DatasetPatch,
    registry: RegistryRepositories = Depends(get_registry),
) -> DatasetResponse:
    return DatasetResponse.from_record(
        registry.datasets.update(dataset_id, **payload.model_dump(exclude_unset=True))
    )


@router.delete("/datasets/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset(dataset_id: int, registry: RegistryRepositories = Depends(get_registry)) -> Response:
    registry.datasets.soft_delete(dataset_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
