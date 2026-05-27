# .latexmkrc — capítulo standalone
# Lee standalone_preamble.tex y thesis-commands.sty desde la carpeta padre
$pdf_mode = 1;
$bibtex_use = 2;  # usar biber

# Archivos auxiliares dentro de aux_files/
$out_dir = 'aux_files';

# TEXINPUTS: buscar .sty también en la carpeta padre (caps_larraga/)
$ENV{TEXINPUTS} = '../:' . ($ENV{TEXINPUTS} // '');

# Tras compilación exitosa: copiar el PDF al directorio del capítulo
$success_cmd = 'cp aux_files/%R.pdf .';

push @generated_exts, 'run.xml', 'bcf', 'bbl', 'lof', 'lot', 'synctex.gz', 'idx', 'ind', 'ilg';
