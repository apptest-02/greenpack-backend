import asyncio  
from app.services.report_service import generate_pdf_report, generate_excel_report  
from app.database import AsyncSessionLocal  
from app.models.base import InspectionJob, InspectionResult  
from sqlalchemy import select  
  
async def fix():  
    async with AsyncSessionLocal() as db:  
        result = await db.execute(select(InspectionJob, InspectionResult).join(InspectionResult, InspectionJob.id == InspectionResult.job_id).where(InspectionJob.id == '4d9f1454-be0a-421e-8257-a0935072dfff'))  
        row = result.first()  
        if row:  
            job, res = row  
            pdf = await generate_pdf_report(job, res)  
            excel = await generate_excel_report(job, res)  
            res.report_pdf_path = pdf  
            res.excel_path = excel  
            await db.commit()  
            print(f'Fixed! PDF: {pdf}, Excel: {excel}')  
        else:  
            print('Job not found')  
  
asyncio.run(fix())  
