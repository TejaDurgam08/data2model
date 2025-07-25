from fpdf import FPDF
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score, accuracy_score

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, ' ML Model Report', ln=True, align='C')
        self.ln(5)

    def section_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 8, f'{title}', ln=True)
        self.ln(2)

    def section_text(self, text):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 6, text)
        self.ln()

    def insert_image(self, path, w=160):
        if os.path.exists(path):
            self.image(path, w=w)
            self.ln()

def sanitize_filename(name):
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', name)

def generate_report(results, output_path="media/report.pdf"):
    os.makedirs("media", exist_ok=True)
    pdf = PDF()
    pdf.add_page()

    for res in results:
        model_name = res["name"]
        safe_name = sanitize_filename(model_name)
        y_test = res["y_test"]
        y_pred = res["y_pred"]
        task = res["task"]

        pdf.section_title(f"{model_name} [{task}]")

        if task == "Classification":
            acc = accuracy_score(y_test, y_pred)
            pdf.section_text(f"Accuracy: {acc*100:.2f}%")

            # Confusion Matrix
            try:
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(4, 3))  # smaller fig size
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                fig_path = f"media/{safe_name}_confusion.png"
                fig.savefig(fig_path, bbox_inches='tight')
                plt.close(fig)
                pdf.insert_image(fig_path, w=140)
            except Exception as e:
                pdf.section_text(f"Error generating confusion matrix: {e}")

        else:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            pdf.section_text(f"R² Score: {r2:.4f}\nMSE: {mse:.4f}")

            try:
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.scatter(y_test, y_pred, alpha=0.6)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                ax.set_title("Actual vs Predicted")
                fig_path = f"media/{safe_name}_scatter.png"
                fig.savefig(fig_path, bbox_inches='tight')
                plt.close(fig)
                pdf.insert_image(fig_path, w=140)
            except Exception as e:
                pdf.section_text(f"Error generating scatter plot: {e}")

    pdf.output(output_path)
    return output_path
