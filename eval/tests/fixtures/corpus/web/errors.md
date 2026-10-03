# Errors

Raise `HTTPError` with a status code and a detail message; the server turns it
into a JSON response with that status.

## Validation errors

A request body that fails validation gets a 422 response that lists every
invalid field.
