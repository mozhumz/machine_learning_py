com_sup="*/1 * * * * /root/anaconda3/bin/python3 supervisor.py"
crontab -l > cron_bp 2>/dev/null
sed -i '/supervisor/d' cron_bp
echo "${com_sup}" >> cron_bp
echo "" >> cron_bp
crontab cron_bp