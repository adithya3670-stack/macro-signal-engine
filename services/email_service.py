import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd
import datetime

class EmailService:
    def __init__(self, sender_email, sender_password):
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587

    def send_portfolio_report(self, recipient_email, profile_name, portfolio_data):
        """
        Sends an HTML email report with portfolio details.
        
        portfolio_data format:
        {
            'metrics': {'Final Balance': '$10,000', 'Total Return': '15%', ...},
            'holdings': [{'Asset': 'SP500', 'Weight': '20%', 'Value': '$2000'}],
            'transactions': [{'Date': '...', 'Asset': '...', 'Action': 'BUY', 'Price': '...'}]
        }
        """
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"🚀 Portfolio Update: {profile_name} - {datetime.date.today()}"
            msg["From"] = self.sender_email
            msg["To"] = recipient_email

            # Create HTML Body
            html_content = self._generate_html(profile_name, portfolio_data)
            msg.attach(MIMEText(html_content, "html"))

            # Send Email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.sender_email, self.sender_password)
            server.sendmail(self.sender_email, recipient_email, msg.as_string())
            server.quit()
            
            return True, "Email sent successfully"
        except Exception as e:
            return False, str(e)

    def _generate_html(self, profile_name, data):
        # Helper to create HTML tables
        def create_table(items, headers=None):
            if not items: return "<p>No data available.</p>"
            
            if not headers:
                headers = items[0].keys()
                
            html = '<table style="width:100%; border-collapse: collapse; margin-bottom: 20px;">'
            html += '<tr style="background-color: #333; color: white;">'
            for h in headers:
                html += f'<th style="padding: 10px; border: 1px solid #ddd;">{h}</th>'
            html += '</tr>'
            
            for item in items:
                html += '<tr>'
                for h in headers:
                    val = item.get(h, '')
                    html += f'<td style="padding: 8px; border: 1px solid #ddd;">{val}</td>'
                html += '</tr>'
            html += '</table>'
            return html

        # Metrics Section
        metrics_html = ""
        if 'metrics' in data and data['metrics']:
            metrics_html = '<div style="display: flex; gap: 20px; margin-bottom: 20px;">'
            for k, v in data['metrics'].items():
                metrics_html += f'''
                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #ddd;">
                    <div style="font-size: 12px; color: #666;">{k}</div>
                    <div style="font-size: 18px; font-weight: bold; color: #2c3e50;">{v}</div>
                </div>
                '''
            metrics_html += '</div>'

        # Actions Section - ALWAYS show this section
        actions_html = ""
        if 'actions' in data:  # Check if key exists (not if list is empty)
            contribution_header = ""
            if data.get('contribution_amount'):
                contribution_header = f'''
                <div style="background: #e3f2fd; padding: 12px; border-radius: 6px; margin-bottom: 15px; text-align: center; border-left: 4px solid #2196f3;">
                    <strong style="color: #1976d2;">💰 Today's Contribution: {data['contribution_amount']}</strong>
                    <div style="font-size: 12px; color: #666; margin-top: 4px;">Distribution based on strategy targets</div>
                </div>
                '''
            
            # Show actions table if there are actions, otherwise show "no rebalancing" message
            if data['actions']:
                actions_table = create_table(data['actions'], headers=['Asset', 'Action', 'Amount', 'Details'])
            else:
                actions_table = '''
                <div style="background: #d4edda; padding: 15px; border-radius: 6px; text-align: center; border-left: 4px solid #28a745;">
                    <strong style="color: #155724;">✓ No rebalancing needed today.</strong>
                    <div style="font-size: 12px; color: #155724; margin-top: 4px;">All position changes are within the trade threshold.</div>
                </div>
                '''
            
            actions_html = f'''
            <h3 style="color: #f39c12;">⚠️ Actions Required Today</h3>
            {contribution_header}
            {actions_table}
            '''

        return f"""
        <html>
        <body style="font-family: Arial, sans-serif; color: #333; line-height: 1.6;">
            <div style="max-width: 800px; margin: 0 auto; padding: 20px;">
                <h2 style="color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px;">
                    Portfolio Report: {profile_name}
                </h2>
                <p>Here is your daily automated portfolio update for <b>{datetime.date.today()}</b>.</p>
                
                <h3 style="color: #e67e22;">📈 Performance Metrics</h3>
                {metrics_html}
                
                <h3 style="color: #27ae60;">💼 Today's Portfolio</h3>
                {create_table(data.get('holdings', []))}
                
                {actions_html}
                
                <h3 style="color: #2980b9;">📝 Recent Transactions (Last 5)</h3>
                {create_table(data.get('transactions', []))}
                
                <hr style="margin-top: 30px; border: 0; border-top: 1px solid #eee;">
                <p style="font-size: 12px; color: #999; text-align: center;">
                    Sent by MacroEconomic Automation • {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                </p>
            </div>
        </body>
        </html>
        """
