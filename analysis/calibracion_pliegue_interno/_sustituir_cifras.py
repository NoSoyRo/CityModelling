"""Sustituye en los .tex las cifras del par publicado por las del pliegue interno.

Cada sustitucion exige que la cadena original aparezca exactamente una vez en el
archivo. Si alguna no aparece, o aparece repetida, el script aborta sin escribir
nada, de modo que no hay cambios silenciosos.

Las cifras provienen de resultados/evaluacion_final_cinco_ventanas.json. Este
script es de un solo uso y se conserva para que el cambio quede auditable.
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
CAPS = RAIZ / "report/tesis/tesis_indice_nuevo/caps_larraga"

CAP06 = CAPS / "cap06_resultados_analisis/cap06_resultados_analisis.tex"
CAP07 = CAPS / "cap07_conclusiones/cap07_conclusiones.tex"
CAP05 = CAPS / "cap05_modelo_crecimiento_urbano/cap05_modelo_crecimiento_urbano.tex"
RESUMEN = CAPS / "front/resumen.tex"

SUSTITUCIONES: dict[Path, list[tuple[str, str]]] = {
    CAP06: [
        # --- parametros del modelo ---
        (r"Umbral de urbanización & $\theta = 0{,}75$ (calibrado sobre la probabilidad combinada). \\",
         r"Umbral de urbanización & $\theta = 0{,}85$ (calibrado sobre la probabilidad combinada). \\"),
        (r"Peso de vecindad & $\alpha = 0{,}50$. \\",
         r"Peso de vecindad & $\alpha = 0{,}60$. \\"),
        (r"Umbral de urbanización & $\theta = 0{,}75$ \\",
         r"Umbral de urbanización & $\theta = 0{,}85$ \\"),
        (r"Peso de vecindad & $\alpha = 0{,}50$ \\",
         r"Peso de vecindad & $\alpha = 0{,}60$ \\"),
        (r"$\alpha = 0{,}50$, el peso de vecindad",
         r"$\alpha = 0{,}60$, el peso de vecindad"),
        (r"supera el umbral $\theta = 0{,}75$ y un sorteo aleatorio lo confirma",
         r"supera el umbral $\theta = 0{,}85$ y un sorteo aleatorio lo confirma"),
        (r"(WoE-AC, umbral $\theta=0{,}75$)", r"(WoE-AC, umbral $\theta=0{,}85$)"),

        # --- tabla de metricas en las cinco ventanas ---
        (r"2011--2016 & $0{,}665$ & $0{,}581$ & $0{,}590$ & $0{,}975$ & $0{,}735$ & $0{,}349$ & $0{,}222$ \\",
         r"2011--2016 & $0{,}713$ & $0{,}614$ & $0{,}631$ & $0{,}958$ & $0{,}761$ & $0{,}438$ & $0{,}240$ \\"),
        (r"2012--2017 & $0{,}750$ & $0{,}652$ & $0{,}688$ & $0{,}925$ & $0{,}789$ & $0{,}499$ & $0{,}345$ \\",
         r"2012--2017 & $0{,}786$ & $0{,}677$ & $0{,}741$ & $0{,}887$ & $0{,}807$ & $0{,}571$ & $0{,}345$ \\"),
        (r"2013--2018 & $0{,}744$ & $0{,}648$ & $0{,}689$ & $0{,}917$ & $0{,}787$ & $0{,}482$ & $0{,}355$ \\",
         r"2013--2018 & $0{,}777$ & $0{,}669$ & $0{,}739$ & $0{,}876$ & $0{,}802$ & $0{,}550$ & $0{,}347$ \\"),
        (r"2014--2019 & $0{,}697$ & $0{,}614$ & $0{,}628$ & $0{,}965$ & $0{,}761$ & $0{,}396$ & $0{,}287$ \\",
         r"2014--2019 & $0{,}745$ & $0{,}649$ & $0{,}675$ & $0{,}942$ & $0{,}787$ & $0{,}490$ & $0{,}307$ \\"),
        (r"2015--2020 & $0{,}755$ & $0{,}669$ & $0{,}701$ & $0{,}936$ & $0{,}802$ & $0{,}498$ & $0{,}378$ \\",
         r"2015--2020 & $0{,}792$ & $0{,}696$ & $0{,}755$ & $0{,}899$ & $0{,}820$ & $0{,}578$ & $0{,}384$ \\"),
        (r"\textbf{Promedio} & $\mathbf{0{,}722}$ & $\mathbf{0{,}633}$ & $\mathbf{0{,}659}$ & $\mathbf{0{,}944}$ & $\mathbf{0{,}775}$ & $\mathbf{0{,}445}$ & $\mathbf{0{,}317}$ \\",
         r"\textbf{Promedio} & $\mathbf{0{,}763}$ & $\mathbf{0{,}661}$ & $\mathbf{0{,}708}$ & $\mathbf{0{,}912}$ & $\mathbf{0{,}795}$ & $\mathbf{0{,}526}$ & $\mathbf{0{,}325}$ \\"),

        # --- tabla comparativa con la literatura ---
        (r"\textbf{FoM promedio} = $\mathbf{0{,}317}$ (i.e., $\mathbf{31{,}7\%}$); \textbf{Kappa promedio} = $\mathbf{0{,}445}$. \\",
         r"\textbf{FoM promedio} = $\mathbf{0{,}325}$ (i.e., $\mathbf{32{,}5\%}$); \textbf{Kappa promedio} = $\mathbf{0{,}526}$. \\"),

        # --- dispersion entre ventanas ---
        (r"el FoM varía en un factor de $1{,}70$ entre el valor mínimo y el máximo, el Kappa en $1{,}43$ y el IoU en $1{,}15$",
         r"el FoM varía en un factor de $1{,}60$ entre el valor mínimo y el máximo, el Kappa en $1{,}32$ y el IoU en $1{,}13$"),
        (r"FoM $=0{,}317$, Kappa $=0{,}445$, IoU $=0{,}633$",
         r"FoM $=0{,}325$, Kappa $=0{,}526$, IoU $=0{,}661$"),

        # --- ventana 2011-2016 ---
        (r"Es el período con el FoM más bajo de la serie ($0{,}222$)",
         r"Es el período con el FoM más bajo de la serie ($0{,}240$)"),
        (r"Accuracy  & $0{,}665$ & FoM                  & \textbf{$0{,}222$} \\",
         r"Accuracy  & $0{,}713$ & FoM                  & \textbf{$0{,}240$} \\"),
        (r"IoU       & $0{,}581$ & Hits ($B$)           & $361\,614$ \\",
         r"IoU       & $0{,}614$ & Hits ($B$)           & $318\,214$ \\"),
        (r"Precision & $0{,}590$ & Falsas alarmas ($A$) & $1\,203\,567$ \\",
         r"Precision & $0{,}631$ & Falsas alarmas ($A$) & $901\,070$ \\"),
        (r"F1        & $0{,}735$ & TP / FP / FN         & $2\,514\,806$ / $1\,749\,463$ / $64\,208$ \\",
         r"F1        & $0{,}761$ & TP / FP / FN         & $2\,471\,406$ / $1\,446\,966$ / $107\,608$ \\"),
        (r"FoM $=0{,}222$; la anomalía se debe a que el área urbana observada disminuyó",
         r"FoM $=0{,}240$; la anomalía se debe a que el área urbana observada disminuyó"),

        # --- ventana 2012-2017 ---
        (r"con un FoM de $0{,}345$ sobre un crecimiento observado positivo",
         r"con un FoM de $0{,}345$ sobre un crecimiento observado positivo"),
        (r"Accuracy  & $0{,}750$ & FoM                  & \textbf{$0{,}345$} \\",
         r"Accuracy  & $0{,}786$ & FoM                  & \textbf{$0{,}345$} \\"),
        (r"IoU       & $0{,}652$ & Hits ($B$)           & $602\,889$ \\",
         r"IoU       & $0{,}677$ & Hits ($B$)           & $498\,699$ \\"),
        (r"Precision & $0{,}688$ & Falsas alarmas ($A$) & $937\,345$ \\",
         r"Precision & $0{,}741$ & Falsas alarmas ($A$) & $639\,088$ \\"),
        (r"F1        & $0{,}789$ & TP / FP / FN         & $2\,532\,942$ / $1\,147\,725$ / $205\,308$ \\",
         r"F1        & $0{,}807$ & TP / FP / FN         & $2\,428\,752$ / $849\,468$ / $309\,498$ \\"),
        (r"$\kappa$  & $0{,}499$ & Crec.\ obs.\ / pred.\ & $+27{,}9\%$ / $+72{,}0\%$ \\",
         r"$\kappa$  & $0{,}571$ & Crec.\ obs.\ / pred.\ & $+27{,}9\%$ / $+53{,}2\%$ \\"),
        (r"Los $602\,889$ hits ($B$) corresponden",
         r"Los $498\,699$ hits ($B$) corresponden"),

        # --- ventana 2013-2018 ---
        (r"que alcanza un FoM de $0{,}355$, el más alto hasta aquí",
         r"que alcanza un FoM de $0{,}347$, el más alto hasta aquí"),
        (r"Accuracy  & $0{,}744$ & FoM                  & \textbf{$0{,}355$} \\",
         r"Accuracy  & $0{,}777$ & FoM                  & \textbf{$0{,}347$} \\"),
        (r"IoU       & $0{,}648$ & Hits ($B$)           & $641\,089$ \\",
         r"IoU       & $0{,}669$ & Hits ($B$)           & $525\,557$ \\"),
        (r"Precision & $0{,}689$ & Falsas alarmas ($A$) & $934\,924$ \\",
         r"Precision & $0{,}739$ & Falsas alarmas ($A$) & $641\,508$ \\"),
        (r"F1        & $0{,}787$ & TP / FP / FN         & $2\,559\,805$ / $1\,156\,835$ / $231\,317$ \\",
         r"F1        & $0{,}802$ & TP / FP / FN         & $2\,444\,273$ / $863\,419$ / $346\,849$ \\"),
        (r"$\kappa$  & $0{,}482$ & Crec.\ obs.\ / pred.\ & $+30{,}4\%$ / $+73{,}6\%$ \\",
         r"$\kappa$  & $0{,}550$ & Crec.\ obs.\ / pred.\ & $+30{,}4\%$ / $+54{,}5\%$ \\"),
        (r"FoM $=0{,}355$; valor cercano al del período 2012--2017",
         r"FoM $=0{,}347$; valor cercano al del período 2012--2017"),
        (r"El número de omisiones ($231\,317$ píxeles)",
         r"El número de omisiones ($346\,849$ píxeles)"),

        # --- ventana 2014-2019 ---
        (r"la Tabla~\ref{tab:metrics_2014_2019} registra un FoM de $0{,}287$",
         r"la Tabla~\ref{tab:metrics_2014_2019} registra un FoM de $0{,}307$"),
        (r"Accuracy  & $0{,}697$ & FoM                  & \textbf{$0{,}287$} \\",
         r"Accuracy  & $0{,}745$ & FoM                  & \textbf{$0{,}307$} \\"),
        (r"IoU       & $0{,}614$ & Hits ($B$)           & $511\,475$ \\",
         r"IoU       & $0{,}649$ & Hits ($B$)           & $449\,249$ \\"),
        (r"Precision & $0{,}628$ & Falsas alarmas ($A$) & $1\,176\,288$ \\",
         r"Precision & $0{,}675$ & Falsas alarmas ($A$) & $856\,224$ \\"),
        (r"F1        & $0{,}761$ & TP / FP / FN         & $2\,611\,594$ / $1\,545\,800$ / $93\,486$ \\",
         r"F1        & $0{,}787$ & TP / FP / FN         & $2\,549\,368$ / $1\,225\,736$ / $155\,712$ \\"),
        (r"$\kappa$  & $0{,}396$ & Crec.\ obs.\ / pred.\ & $+9{,}5\%$ / $+68{,}3\%$ \\",
         r"$\kappa$  & $0{,}490$ & Crec.\ obs.\ / pred.\ & $+9{,}5\%$ / $+52{,}9\%$ \\"),
        (r"FoM $=0{,}287$; el crecimiento real modesto ($+9{,}5\%$) contrasta con una predicción de expansión más elevada ($+68{,}3\%$), lo que se refleja en la proporción de falsas alarmas ($1\,176\,288$ píxeles). El Recall alto ($0{,}965$)",
         r"FoM $=0{,}307$; el crecimiento real modesto ($+9{,}5\%$) contrasta con una predicción de expansión más elevada ($+52{,}9\%$), lo que se refleja en la proporción de falsas alarmas ($856\,224$ píxeles). El Recall alto ($0{,}942$)"),

        # --- ventana 2015-2020 ---
        (r"La Tabla~\ref{tab:metrics_2015_2020} consigna un FoM de $0{,}378$",
         r"La Tabla~\ref{tab:metrics_2015_2020} consigna un FoM de $0{,}384$"),
        (r"Accuracy  & $0{,}755$ & FoM                  & \textbf{$0{,}378$} \\",
         r"Accuracy  & $0{,}792$ & FoM                  & \textbf{$0{,}384$} \\"),
        (r"IoU       & $0{,}669$ & Hits ($B$)           & $704\,983$ \\",
         r"IoU       & $0{,}696$ & Hits ($B$)           & $597\,253$ \\"),
        (r"Precision & $0{,}701$ & Falsas alarmas ($A$) & $977\,214$ \\",
         r"Precision & $0{,}755$ & Falsas alarmas ($A$) & $669\,051$ \\"),
        (r"Recall    & $0{,}936$ & Omisiones ($C$)      & $181\,887$ \\",
         r"Recall    & $0{,}899$ & Omisiones ($C$)      & $289\,617$ \\"),
        (r"F1        & $0{,}802$ & TP / FP / FN         & $2\,682\,235$ / $1\,145\,318$ / $181\,887$ \\",
         r"F1        & $0{,}820$ & TP / FP / FN         & $2\,574\,505$ / $837\,155$ / $289\,617$ \\"),
        (r"$\kappa$  & $0{,}498$ & Crec.\ obs.\ / pred.\ & $+33{,}5\%$ / $+78{,}4\%$ \\",
         r"$\kappa$  & $0{,}578$ & Crec.\ obs.\ / pred.\ & $+33{,}5\%$ / $+59{,}0\%$ \\"),
        (r"FoM $=0{,}378$; entre las cinco ventanas es el FoM más alto, en un contexto de fuerte crecimiento observado ($+33{,}5\%$). Se registran 704\,983 hits",
         r"FoM $=0{,}384$; entre las cinco ventanas es el FoM más alto, en un contexto de fuerte crecimiento observado ($+33{,}5\%$). Se registran 597\,253 hits"),
        (r"Mapa de error para la ventana 2015$\rightarrow$2020 (FoM $=0{,}378$)",
         r"Mapa de error para la ventana 2015$\rightarrow$2020 (FoM $=0{,}384$)"),

        # --- tabla de Pontius ---
        (r"2011--2016 & $0{,}222$ & $31{,}1\%$ & $2{,}4\%$  & $92{,}9\%$ \\",
         r"2011--2016 & $0{,}240$ & $24{,}7\%$ & $4{,}0\%$  & $86{,}2\%$ \\"),
        (r"2012--2017 & $0{,}345$ & $17{,}4\%$ & $7{,}6\%$  & $69{,}7\%$ \\",
         r"2012--2017 & $0{,}345$ & $10{,}0\%$ & $11{,}4\%$ & $46{,}6\%$ \\"),
        (r"2013--2018 & $0{,}355$ & $17{,}1\%$ & $8{,}5\%$  & $66{,}7\%$ \\",
         r"2013--2018 & $0{,}347$ & $9{,}5\%$  & $12{,}8\%$ & $42{,}7\%$ \\"),
        (r"2014--2019 & $0{,}287$ & $26{,}8\%$ & $3{,}5\%$  & $88{,}6\%$ \\",
         r"2014--2019 & $0{,}307$ & $19{,}7\%$ & $5{,}7\%$  & $77{,}5\%$ \\"),
        (r"2015--2020 & $0{,}378$ & $17{,}8\%$ & $6{,}7\%$  & $72{,}6\%$ \\",
         r"2015--2020 & $0{,}384$ & $10{,}1\%$ & $10{,}7\%$ & $48{,}6\%$ \\"),
        (r"\textbf{Promedio} & $\mathbf{0{,}317}$ & $\mathbf{22{,}0\%}$ & $\mathbf{5{,}7\%}$ & $\mathbf{79{,}4\%}$ \\",
         r"\textbf{Promedio} & $\mathbf{0{,}325}$ & $\mathbf{14{,}8\%}$ & $\mathbf{8{,}9\%}$ & $\mathbf{62{,}4\%}$ \\"),

        # --- hipotesis y cierre ---
        (r"pero el FoM varía entre $0{,}222$ y $0{,}378$ (Tabla~\ref{tab:metrics_all_periods})",
         r"pero el FoM varía entre $0{,}240$ y $0{,}384$ (Tabla~\ref{tab:metrics_all_periods})"),
        (r"El FoM promedio de $0{,}317$ se ubica en la banda intermedia",
         r"El FoM promedio de $0{,}325$ se ubica en la banda intermedia"),
        (r"El Kappa de $0{,}445$ se interpreta entonces frente a la escala de \textcite{LandisKoch1977}, y el IoU de $0{,}633$ mide",
         r"El Kappa de $0{,}526$ se interpreta entonces frente a la escala de \textcite{LandisKoch1977}, y el IoU de $0{,}661$ mide"),
        (r"(promedio $0{,}317$, entre $0{,}222$ y $0{,}378$)",
         r"(promedio $0{,}325$, entre $0{,}240$ y $0{,}384$)"),
        (r"y donde el promedio alcanzado es $0{,}317$",
         r"y donde el promedio alcanzado es $0{,}325$"),
        (r"Su FoM promedio de $0{,}317$ se interpreta frente al espectro multi-sitio",
         r"Su FoM promedio de $0{,}325$ se interpreta frente al espectro multi-sitio"),
    ],
}


def main() -> None:
    fallos: list[str] = []
    planes: list[tuple[Path, str]] = []

    for ruta, pares in SUSTITUCIONES.items():
        texto = ruta.read_text()
        for viejo, nuevo in pares:
            n = texto.count(viejo)
            if n != 1:
                fallos.append(f"{ruta.name}: {n} coincidencias de «{viejo[:70]}…»")
                continue
            texto = texto.replace(viejo, nuevo)
        planes.append((ruta, texto))

    if fallos:
        print(f"ABORTA, {len(fallos)} cadenas no coinciden exactamente una vez:")
        for f in fallos:
            print("  " + f)
        sys.exit(1)

    for ruta, texto in planes:
        ruta.write_text(texto)
        print(f"  escrito {ruta.name}")


if __name__ == "__main__":
    main()
