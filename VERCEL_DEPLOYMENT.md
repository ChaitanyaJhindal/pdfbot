# Vercel Deployment Guide for PDF Chatbot API

## Prerequisites

1. **Vercel Account**: Sign up at [vercel.com](https://vercel.com)
2. **GitHub Repository**: Push your code to GitHub
3. **Environment Variables**: Have your API keys ready

## Deployment Steps

### 1. Install Vercel CLI (Optional)

```bash
npm install -g vercel
```

### 2. Connect GitHub Repository to Vercel

1. Go to [vercel.com/dashboard](https://vercel.com/dashboard)
2. Click "New Project"
3. Import your GitHub repository
4. Vercel will auto-detect it as a Python project

### 3. Configure Environment Variables

In your Vercel project dashboard, go to Settings > Environment Variables and add:

```
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
HACKRX_API_KEY=your_hackrx_api_key_here
```

### 4. Deploy

- **Automatic**: Push to your main branch, Vercel will auto-deploy
- **Manual**: Use Vercel CLI: `vercel --prod`

### 5. Test Your Deployment

Your API will be available at: `https://your-project-name.vercel.app`

Test endpoints:
- Health check: `GET https://your-project-name.vercel.app/`
- HackRx endpoint: `POST https://your-project-name.vercel.app/hackrx/run`

## Important Notes for Vercel

### Limitations
- **Cold starts**: First request may be slower
- **Timeout**: 300 seconds max execution time
- **Memory**: 1024MB limit
- **File storage**: Temporary only, files are cleaned up

### Optimizations Made
- Reduced logging verbosity
- No persistent file caching
- Automatic temp file cleanup
- Question limit (5 per request)
- Shorter request timeouts

### Environment Variables Required
```
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
HACKRX_API_KEY=your_hackrx_key
```

## Testing Your Deployed API

### Health Check
```bash
curl https://your-project-name.vercel.app/
```

### HackRx Endpoint Test
```bash
curl -X POST "https://your-project-name.vercel.app/hackrx/run" \
  -H "Authorization: Bearer your_hackrx_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.com/document.pdf",
    "questions": ["What is this document about?"]
  }'
```

## Troubleshooting

### Common Issues
1. **Function timeout**: Reduce PDF size or number of questions
2. **Memory limit**: Process smaller documents
3. **Cold start delays**: First request takes longer

### Monitoring
- Check Vercel Functions logs in the dashboard
- Monitor function execution time and memory usage

## Files Structure for Vercel
```
├── main.py              # Entry point for Vercel
├── api_vercel.py        # Vercel-optimized API
├── advanced_pdf_bot.py  # Core chatbot logic
├── vercel.json          # Vercel configuration
├── requirements-vercel.txt # Python dependencies
├── .vercelignore        # Files to ignore
└── .env.template        # Environment variables template
```
