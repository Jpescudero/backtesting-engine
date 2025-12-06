#!/usr/bin/env bash
#
# Copia todos los scripts (.py) listados en backtesting_project_structure.txt
# desde tu proyecto local (SRC_ROOT) a:
#   C:\Users\JorgeP\Desktop\Scripts
# respetando la estructura de carpetas.
#
# Este .sh está pensado para estar en:
#   C:\Users\JorgeP\OneDrive\Bolsa\Scripts\10. Backtesting
#
# Uso típico (en Git Bash):
#   cd /c/RUTA/A/TU/backtesting-engine   # raíz del proyecto
#   "/c/Users/JorgeP/OneDrive/Bolsa/Scripts/10. Backtesting/copy_from_structure.sh"
#
# Opcionalmente:
#   copy_from_structure.sh [STRUCT_FILE] [SRC_ROOT] [DEST_ROOT]
#
# Donde:
#   STRUCT_FILE -> por defecto el .txt que está en la misma carpeta que este .sh
#   SRC_ROOT    -> por defecto el directorio actual (pwd)
#   DEST_ROOT   -> por defecto /c/Users/JorgeP/Desktop/Scripts

set -euo pipefail

# Carpeta donde está este script (.sh)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Fichero de estructura por defecto: el que está junto al script
STRUCT_FILE="${1:-${SCRIPT_DIR}/backtesting_project_structure.txt}"

# Raíz del proyecto local (donde está main.py, src/, etc.)
SRC_ROOT="${2:-$(pwd)}"

# Carpeta destino fija en tu Escritorio
DEST_ROOT="${3:-/c/Users/JorgeP/Desktop/Scripts}"

if [[ ! -f "$STRUCT_FILE" ]]; then
  echo "ERROR: no encuentro el fichero de estructura: $STRUCT_FILE" >&2
  exit 1
fi
#!/usr/bin/env bash
#
# Copia todos los scripts (.py) listados en backtesting_project_structure.txt
# desde tu proyecto local (SRC_ROOT) a:
#   C:\Users\JorgeP\Desktop\Scripts
# respetando la estructura de carpetas
# y luego genera:
#   C:\Users\JorgeP\Desktop\Scripts.zip
#
# Este .sh está pensado para estar en:
#   C:\Users\JorgeP\OneDrive\Bolsa\Scripts\10. Backtesting
#
# Uso típico (en Git Bash):
#   cd /c/RUTA/A/TU/backtesting-engine   # raíz del proyecto
#   "/c/Users/JorgeP/OneDrive/Bolsa/Scripts/10. Backtesting/copy_from_structure.sh"
#
# Opcionalmente:
#   copy_from_structure.sh [STRUCT_FILE] [SRC_ROOT] [DEST_ROOT]
#
# Donde:
#   STRUCT_FILE -> por defecto el .txt que está en la misma carpeta que este .sh
#   SRC_ROOT    -> por defecto el directorio actual (pwd)
#   DEST_ROOT   -> por defecto /c/Users/JorgeP/Desktop/Scripts

set -euo pipefail

# Carpeta donde está este script (.sh)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Fichero de estructura por defecto: el que está junto al script
STRUCT_FILE="${1:-${SCRIPT_DIR}/backtesting_project_structure.txt}"

# Raíz del proyecto local (donde está main.py, src/, etc.)
SRC_ROOT="${2:-$(pwd)}"

# Carpeta destino fija en tu Escritorio
DEST_ROOT="${3:-/c/Users/JorgeP/Desktop/Scripts}"

# Ruta del ZIP que queremos generar
ZIP_PATH="/c/Users/JorgeP/Desktop/Scripts.zip"

if [[ ! -f "$STRUCT_FILE" ]]; then
  echo "ERROR: no encuentro el fichero de estructura: $STRUCT_FILE" >&2
  exit 1
fi

echo "Usando estructura: $STRUCT_FILE"
echo "Repo local (SRC_ROOT): $SRC_ROOT"
echo "Carpeta destino (DEST_ROOT): $DEST_ROOT"
echo "ZIP final: $ZIP_PATH"
echo

# Creamos la carpeta destino base por si no existe
mkdir -p "$DEST_ROOT"

# Extrae todas las URLs de GitHub con blob/main (ficheros concretos)
# y elimina duplicados
grep -o 'https://github.com[^ ]*backtesting-engine/blob/main[^ ]*' "$STRUCT_FILE" \
  | sort -u \
  | while read -r url; do

    # Ruta relativa dentro del repo (después de 'blob/main/')
    rel_path="${url#*blob/main/}"

    src_path="${SRC_ROOT}/${rel_path}"
    dest_path="${DEST_ROOT}/${rel_path}"
    dest_dir="$(dirname "$dest_path")"

    echo "Procesando: $rel_path"
    echo "  origen:  $src_path"
    echo "  destino: $dest_path"

    if [[ ! -f "$src_path" ]]; then
      echo "  [AVISO] El fichero de origen NO existe en SRC_ROOT, se omite."
      echo
      continue
    fi

    mkdir -p "$dest_dir"
    cp "$src_path" "$dest_path"
    echo "  Copiado."
    echo
  done

echo "Copia de scripts completada."
echo

# --- Crear ZIP con todo el contenido de DEST_ROOT ---

# Borramos ZIP anterior si existe
if [[ -f "$ZIP_PATH" ]]; then
  echo "Eliminando ZIP anterior: $ZIP_PATH"
  rm -f "$ZIP_PATH"
fi

echo "Creando ZIP con el contenido de: $DEST_ROOT"
(
  cd "$DEST_ROOT"
  # -r: recursivo, . -> todo el contenido de DEST_ROOT
  zip -r "$ZIP_PATH" .
)

echo
echo "Fin."
echo "Todos los scripts listados se han copiado a: $DEST_ROOT"
echo "ZIP generado en: $ZIP_PATH"
