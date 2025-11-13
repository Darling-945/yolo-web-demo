# 部署指南

## 环境要求

### 系统要求
- **操作系统**: Linux (推荐 Ubuntu 20.04+), macOS, Windows 10+
- **CPU**: x86_64 架构，推荐 4 核心以上
- **内存**: 最低 4GB，推荐 8GB+
- **存储**: 最低 10GB 可用空间
- **网络**: 稳定的网络连接（用于模型下载）

### 软件依赖
- **Python**: 3.8+ (推荐 3.9+)
- **pip**: 最新版本
- **Git**: 用于代码管理
- **可选**: CUDA Toolkit (GPU加速)

## 快速部署

### 1. 克隆项目

```bash
git clone https://github.com/your-repo/yolo-web-demo.git
cd yolo-web-demo
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 启动服务

```bash
python run.py
```

### 4. 访问应用

- **本地访问**: http://localhost:5000
- **局域网访问**: http://your-ip:5000

## 生产环境部署

### 1. 使用 Gunicorn (推荐)

#### 安装 Gunicorn
```bash
pip install gunicorn
```

#### 创建 Gunicorn 配置文件
```python
# gunicorn_config.py
import multiprocessing

# 服务器配置
bind = "0.0.0.0:5000"
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "gevent"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50

# 超时设置
timeout = 120
keepalive = 2

# 日志配置
accesslog = "logs/access.log"
errorlog = "logs/error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# 安全配置
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190
```

#### 启动命令
```bash
gunicorn -c gunicorn_config.py app:app
```

### 2. 使用 Docker

#### 创建 Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 创建必要目录
RUN mkdir -p static/uploads static/outputs logs models

# 设置环境变量
ENV FLASK_ENV=production
ENV HOST=0.0.0.0
ENV PORT=5000

# 暴露端口
EXPOSE 5000

# 启动命令
CMD ["gunicorn", "-c", "gunicorn_config.py", "app:app"]
```

#### 创建 docker-compose.yml
```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./static/uploads:/app/static/uploads
      - ./static/outputs:/app/static/outputs
      - ./logs:/app/logs
      - ./models:/app/models
      - ./.env:/app/.env
    environment:
      - FLASK_ENV=production
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - web
    restart: unless-stopped

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  redis_data:
```

#### 启动 Docker 服务
```bash
docker-compose up -d
```

### 3. 使用 Systemd 服务

#### 创建服务文件
```ini
# /etc/systemd/system/yolo-web-demo.service
[Unit]
Description=YOLO Web Demo
After=network.target

[Service]
Type=notify
User=www-data
Group=www-data
WorkingDirectory=/var/www/yolo-web-demo
Environment=PATH=/var/www/yolo-web-demo/venv/bin
ExecStart=/var/www/yolo-web-demo/venv/bin/gunicorn -c gunicorn_config.py app:app
ExecReload=/bin/kill -s HUP $MAINPID
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

#### 启用和启动服务
```bash
sudo systemctl daemon-reload
sudo systemctl enable yolo-web-demo
sudo systemctl start yolo-web-demo
sudo systemctl status yolo-web-demo
```

## 反向代理配置

### Nginx 配置

```nginx
# /etc/nginx/sites-available/yolo-web-demo
server {
    listen 80;
    server_name your-domain.com;

    # 重定向到 HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    # SSL 配置
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;

    # 安全头
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    # 文件上传大小限制
    client_max_body_size 50M;

    # 静态文件
    location /static/ {
        alias /var/www/yolo-web-demo/static/;
        expires 1d;
        add_header Cache-Control "public, immutable";
    }

    # API 接口
    location /api/ {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # 超时设置
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # Web 页面
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Apache 配置

```apache
# /etc/apache2/sites-available/yolo-web-demo.conf
<VirtualHost *:80>
    ServerName your-domain.com
    Redirect permanent / https://your-domain.com/
</VirtualHost>

<VirtualHost *:443>
    ServerName your-domain.com

    # SSL 配置
    SSLEngine on
    SSLCertificateFile /etc/ssl/certs/cert.pem
    SSLCertificateKeyFile /etc/ssl/private/key.pem

    # 安全头
    Header always set X-Frame-Options DENY
    Header always set X-Content-Type-Options nosniff
    Header always set X-XSS-Protection "1; mode=block"

    # 文件上传大小限制
    LimitRequestBody 52428800

    # 反向代理配置
    ProxyPreserveHost On
    ProxyRequests Off

    ProxyPass /static/ !
    ProxyPass / http://127.0.0.1:5000/
    ProxyPassReverse / http://127.0.0.1:5000/

    # 静态文件
    DocumentRoot /var/www/yolo-web-demo/static
    <Directory /var/www/yolo-web-demo/static>
        Options -Indexes
        Require all granted
    </Directory>
</VirtualHost>
```

## 环境配置

### 生产环境配置

```bash
# .env
FLASK_ENV=production
SECRET_KEY=your-super-secret-key-here
HOST=0.0.0.0
PORT=5000
MAX_CONTENT_LENGTH=50485760
UPLOAD_FOLDER=static/uploads
OUTPUT_FOLDER=static/outputs
MAX_FILE_AGE=3600
CLEANUP_INTERVAL=300
DEFAULT_MODEL=yolo11n.pt
DEFAULT_CONFIDENCE=0.25
DEFAULT_IOU=0.45
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
RATELIMIT_DEFAULT=1000 per hour
RATELIMIT_API=60 per minute
```

### 开发环境配置

```bash
# .env.development
FLASK_ENV=development
SECRET_KEY=dev-secret-key
HOST=127.0.0.1
PORT=5000
DEBUG=True
LOG_LEVEL=DEBUG
```

## 监控和日志

### 1. 应用监控

#### Prometheus 配置
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'yolo-web-demo'
    static_configs:
      - targets: ['localhost:5000']
    metrics_path: '/api/metrics'
    scrape_interval: 30s
```

#### 健康检查
```bash
# 添加到 crontab
*/5 * * * * curl -f http://localhost:5000/api/health || systemctl restart yolo-web-demo
```

### 2. 日志管理

#### Logrotate 配置
```bash
# /etc/logrotate.d/yolo-web-demo
/var/www/yolo-web-demo/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 www-data www-data
    postrotate
        systemctl reload yolo-web-demo
    endscript
}
```

#### 日志聚合 (ELK Stack)
```yaml
# docker-compose.elk.yml
version: '3.8'

services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:7.14.0
    environment:
      - discovery.type=single-node
    volumes:
      - es_data:/usr/share/elasticsearch/data
    ports:
      - "9200:9200"

  logstash:
    image: docker.elastic.co/logstash/logstash:7.14.0
    volumes:
      - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf
    ports:
      - "5044:5044"
    depends_on:
      - elasticsearch

  kibana:
    image: docker.elastic.co/kibana/kibana:7.14.0
    ports:
      - "5601:5601"
    environment:
      ELASTICSEARCH_URL: http://elasticsearch:9200
    depends_on:
      - elasticsearch

volumes:
  es_data:
```

## 性能优化

### 1. 系统级优化

```bash
# 增加文件描述符限制
echo "* soft nofile 65536" >> /etc/security/limits.conf
echo "* hard nofile 65536" >> /etc/security/limits.conf

# 优化网络参数
echo "net.core.somaxconn = 65535" >> /etc/sysctl.conf
echo "net.ipv4.tcp_max_syn_backlog = 65535" >> /etc/sysctl.conf
sysctl -p
```

### 2. 应用级优化

#### 缓存配置
```python
# 在 app.py 中添加 Redis 缓存
from flask_caching import Cache

cache = Cache(app, config={
    'CACHE_TYPE': 'redis',
    'CACHE_REDIS_URL': 'redis://localhost:6379/0',
    'CACHE_DEFAULT_TIMEOUT': 300
})

@app.cache.cached(timeout=300, key_prefix='models')
def get_models_cached():
    return get_available_models()
```

#### 连接池配置
```python
# 数据库连接池
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True
)
```

## 安全加固

### 1. 防火墙配置

```bash
# UFW 配置
sudo ufw enable
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw deny 5000/tcp  # 只允许内部访问
```

### 2. SSL/TLS 证书

#### 使用 Let's Encrypt
```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

### 3. 安全扫描

```bash
# 安装安全扫描工具
pip install safety bandit

# 扫描依赖漏洞
safety check

# 扫描代码安全问题
bandit -r .
```

## 故障排除

### 常见问题

1. **端口占用**
   ```bash
   sudo netstat -tlnp | grep :5000
   sudo kill -9 <PID>
   ```

2. **权限问题**
   ```bash
   sudo chown -R www-data:www-data /var/www/yolo-web-demo
   sudo chmod -R 755 /var/www/yolo-web-demo
   ```

3. **内存不足**
   ```bash
   # 检查内存使用
   free -h

   # 添加交换空间
   sudo fallocate -l 2G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

4. **GPU 相关问题**
   ```bash
   # 检查 CUDA
   nvidia-smi

   # 安装 PyTorch with CUDA
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### 日志分析

```bash
# 查看应用日志
tail -f logs/app.log

# 查看 Nginx 日志
tail -f /var/log/nginx/access.log
tail -f /var/log/nginx/error.log

# 查看系统日志
journalctl -u yolo-web-demo -f
```

## 备份和恢复

### 1. 数据备份

```bash
#!/bin/bash
# backup.sh
BACKUP_DIR="/backup/yolo-web-demo"
DATE=$(date +%Y%m%d_%H%M%S)

# 创建备份目录
mkdir -p $BACKUP_DIR

# 备份配置文件
tar -czf $BACKUP_DIR/config_$DATE.tar.gz .env models/

# 备份上传的文件
tar -czf $BACKUP_DIR/uploads_$DATE.tar.gz static/uploads/

# 备份日志
tar -czf $BACKUP_DIR/logs_$DATE.tar.gz logs/

# 清理旧备份（保留7天）
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete
```

### 2. 自动备份

```bash
# 添加到 crontab
0 2 * * * /path/to/backup.sh
```

### 3. 恢复流程

```bash
#!/bin/bash
# restore.sh
BACKUP_FILE=$1

# 停止服务
sudo systemctl stop yolo-web-demo

# 恢复文件
tar -xzf $BACKUP_FILE -C /

# 修复权限
sudo chown -R www-data:www-data /var/www/yolo-web-demo

# 启动服务
sudo systemctl start yolo-web-demo
```