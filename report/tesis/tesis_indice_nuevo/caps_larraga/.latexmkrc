# .latexmkrc — configuración de latexmk para la tesis WoE-AC
# La tesis usa biblatex + biber (no bibtex)

$pdf_mode = 1;
$bibtex_use = 2;  # 2 = usar biber

push @generated_exts, 'run.xml', 'bcf', 'bbl', 'lof', 'lot', 'synctex.gz';
