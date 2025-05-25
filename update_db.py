from app import app, db
 
# 在应用上下文中运行数据库更新
with app.app_context():
    db.create_all()
    print("数据库结构已更新!") 