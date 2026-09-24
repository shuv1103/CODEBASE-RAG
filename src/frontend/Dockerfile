# Stage 1: type-check + build the static assets.
# VITE_API_BASE_URL is baked into the JS bundle at build time (Vite inlines
# import.meta.env.* at build, not at container runtime) — it must be the
# URL the *browser* can reach, not an internal Docker service name like
# "api". Default matches docker-compose.yml's api service port mapping.
FROM node:20-alpine AS builder
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
ARG VITE_API_BASE_URL=http://localhost:8000
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL
RUN npm run build

# Stage 2: serve the built assets with nginx.
FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
