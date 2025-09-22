import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt

# Load original DataFrame from URL
url = "https://raw.githubusercontent.com/MaricelSantos/Mentoria--Diplodatos-2025/main/Conexiones_Transparentes.csv"
df_original = pd.read_csv(url)

# Plot the missing data matrix
msno.matrix(df_original, figsize=(15, 5), color=(0, 0, 0.7))

# Save the plot locally
plt.savefig("missing_data_matrix_original.png", dpi=300, bbox_inches='tight')

# Load your DataFrame (example: CSV)
df = pd.read_csv("aguas_limpieza_final.csv")

# Plot the missing data matrix
msno.matrix(df, figsize=(15, 5), color=(0, 0, 0.7))  # You can adjust the color (R,G,B)

# Save the plot to an image file
plt.savefig("missing_data_matrix_limpieza.png", dpi=300, bbox_inches='tight')  # You can change the path/filename

