# Text Embedding Service

A FastAPI-based service that generates text embeddings with customizable dimensions using various transformer models.

## Features

- Basic HTTP Authentication
- Multiple embedding dimension options (384 or 768 dimensions)
- Health check endpoint
- Support for batch text processing
- Environment variable configuration
- Multiple embedding generation methods:
  - Fixed dimension embeddings using different models
  - Dimension reduction using PCA
  - Custom dimension reduction using linear transformation

## Prerequisites

- Python 3.7+
- pip

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root:
```env
API_USERNAME=your_username
API_PASSWORD=your_password
```

## Running the Service

Start the server with:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

Or run directly with Python:
```bash
python main.py
```

## API Endpoints

### Generate Embeddings (384 dimensions)
```http
POST /embed
```
Generates 384-dimensional embeddings using the all-MiniLM-L6-v2 model.

### Generate Embeddings (768 dimensions)
```http
POST /embed2
```
Generates 768-dimensional embeddings using the all-mpnet-base-v2 model.

### Health Check
```http
GET /health
```
Returns service health status and model information.

## Request/Response Format

### Request
```json
{
    "texts": ["Your text here", "Another text"]
}
```

### Response
```json
{
    "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
    "dimensions": 384
}
```

## Authentication

The service uses HTTP Basic Authentication. Include your credentials in the request headers:
```
Authorization: Basic <base64-encoded-credentials>
```

## Models Used

- all-MiniLM-L6-v2 (384 dimensions)
- all-mpnet-base-v2 (768 dimensions)
- paraphrase-MiniLM-L3-v2 (384 dimensions)

## Environment Variables

| Variable | Description |
|----------|-------------|
| API_USERNAME | Username for Basic Auth |
| API_PASSWORD | Password for Basic Auth |

## Project Structure

```
.
├── main.py           # FastAPI application and endpoints
├── embeddings.py     # Embedding generation logic
├── requirements.txt  # Project dependencies
├── .env             # Environment variables
└── README.md        # Project documentation
```

## Error Handling

The service returns appropriate HTTP status codes:
- 401: Invalid credentials
- 500: Internal server error with error details

## Buidling docker image

`docker build -t embedding-service .`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Your License Here]