# Migration Guide

## v2.0 - LiteLLM Migration

### Model Name Format Change

The model naming convention has changed with the migration from `together` to `litellm`:

**Old format:**
```python
model_name = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
```

**New format:**
```python
model_name = "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo"
```

#### Migration Steps

1. **Update Configuration Files**: Add the provider prefix `together_ai/` to all model names in your configuration files.

2. **Update Saved Experiment Results**: If you have saved experiment results or configurations that reference the old model names, you'll need to update them manually or create a script to update the model name fields.

3. **Environment Variables**: Ensure your `.env` file has the correct API keys for the provider you're using:
   ```
   TOGETHER_API_KEY=your_key_here
   ```

#### Supported Providers

With `litellm`, you can now use models from various providers by using the appropriate prefix:
- `together_ai/` - Together AI models
- `openai/` - OpenAI models
- `anthropic/` - Anthropic models
- And many more...

See the [litellm documentation](https://docs.litellm.ai/docs/providers) for a complete list of supported providers.

### Cost Tracking

The LLM interface now includes cost tracking for all API calls. The cost information is logged but not exposed in the public API to maintain backward compatibility. Future versions may expose cost tracking through optional parameters or separate methods.
