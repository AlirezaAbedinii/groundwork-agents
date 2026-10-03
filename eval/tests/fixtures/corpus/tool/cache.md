# Caching

The tool keeps downloaded archives in a cache directory, so a repeated
install never fetches the same archive twice.

## Cache location

Set `TOOL_CACHE_DIR` to move the cache, or pass `--cache-dir` on the
command line.

```bash
# remove every cached archive
tool cache clean
```

## Clearing the cache

Run `tool cache clean` to remove every cached archive.
