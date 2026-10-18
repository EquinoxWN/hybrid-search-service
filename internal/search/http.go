package search

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"strconv"
	"time"
)

// Pinger reports whether the backend is reachable.
type Pinger interface {
	Ping(ctx context.Context) error
}

// Handler serves GET /search?q=...&mode=hybrid|keyword|vector&k=10 and GET /healthz.
func Handler(s *Service, db Pinger, log *slog.Logger) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /search", func(w http.ResponseWriter, r *http.Request) {
		q := r.URL.Query()
		mode, err := ParseMode(q.Get("mode"))
		if err != nil {
			writeError(w, http.StatusBadRequest, err.Error())
			return
		}
		k := 10
		if v := q.Get("k"); v != "" {
			if k, err = strconv.Atoi(v); err != nil {
				writeError(w, http.StatusBadRequest, "bad request: k must be an integer")
				return
			}
		}
		ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
		defer cancel()
		resp, err := s.Search(ctx, q.Get("q"), mode, k)
		switch {
		case errors.Is(err, ErrBadRequest):
			writeError(w, http.StatusBadRequest, err.Error())
		case err != nil:
			log.Error("search failed", "mode", mode, "err", err)
			writeError(w, http.StatusInternalServerError, "search failed")
		default:
			writeJSON(w, http.StatusOK, resp)
		}
	})
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
		defer cancel()
		if err := db.Ping(ctx); err != nil {
			writeError(w, http.StatusServiceUnavailable, "database unreachable")
			return
		}
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})
	return mux
}

// writeJSON sends a JSON body.
func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

// writeError sends {"error": msg}.
func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}
