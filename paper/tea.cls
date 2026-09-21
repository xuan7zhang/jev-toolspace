% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A Green-Themed Academic Style for TEA Preprints
% Author: TEA
% Version: 2.2-double-footnote-fix
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ============================================================================
% Class declarations
% ============================================================================

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{tea}
\LoadClassWithOptions{article}

% ============================================================================
% Layout
% ============================================================================

\RequirePackage[
  top=2.25cm,
  bottom=2.5cm,
  left=2.5cm,
  right=2.5cm,
  columnsep=0.65cm
]{geometry}

\RequirePackage{microtype}
\RequirePackage{placeins}
\RequirePackage{hyphenat}
\RequirePackage{setspace}
\RequirePackage{parskip}
\RequirePackage[latin,english]{babel}
\RequirePackage{etoolbox}

% ============================================================================
% Meta custom fonts
% ============================================================================

\RequirePackage[T1]{fontenc}
\RequirePackage{newtxmath}
\renewcommand*\ttdefault{cmvtt}

\ifdefined\tealocal
  % local TeX Live 2021 cannot embed Optimistic.ttf (OS/2 table v4); Helvetica stands in
  \RequirePackage[scaled=0.9]{helvet}
\else
\DeclareFontFamily{T1}{optimistic}{}
\DeclareFontShape{T1}{optimistic}{m}{n}{<-> s * [0.88] assets/optimistic}{}
\pdfmapline{+optimistic < assets/Optimistic.ttf <T1-WGL4.enc}
\renewcommand{\sfdefault}{optimistic}
\fi
\renewcommand{\familydefault}{\rmdefault} 

% ============================================================================
% Graphics, Tables, Colors
% ============================================================================

\RequirePackage{graphicx}
\RequirePackage{subcaption}
\RequirePackage{booktabs}
\RequirePackage{nicematrix}
\RequirePackage{multirow}
\RequirePackage{bm}
\RequirePackage{xcolor}

\definecolor{accent}{HTML}{2F6F55}
\definecolor{accent2}{HTML}{6FA287}
\definecolor{fg}{HTML}{1F2933}
\definecolor{bg}{HTML}{F4F7F5}

% ============================================================================
% Hyperlinks & References
% ============================================================================

\RequirePackage{hyperref}
\RequirePackage[noabbrev,nameinlink]{cleveref}

\AtBeginDocument{
  \hypersetup{
    colorlinks=true,
    linkcolor=accent,
    citecolor=accent,
    urlcolor=accent
  }
}

\RequirePackage[round,authoryear]{natbib}
\def\bibfont{\small}

% ============================================================================
% Section and caption format
% ============================================================================

\RequirePackage{titlesec}
\titleformat*{\paragraph}{\itshape}
\titleformat*{\section}{\Large\sffamily\bfseries}
\titleformat*{\subsection}{\large\sffamily\bfseries}
\titleformat*{\subsubsection}{\normalsize\sffamily\bfseries}
\DeclareTextFontCommand{\textbf}{\bfseries\sffamily}

\RequirePackage{caption}
\DeclareCaptionLabelSeparator{custom}{}
\DeclareCaptionFormat{custom}{{\sffamily\textbf{#1 #2}} #3}
\captionsetup{singlelinecheck=false,format=custom,labelsep=custom,font=small}
\captionsetup[sub]{singlelinecheck=true,format=custom,labelsep=custom,font=small}

% ============================================================================
% Author / Affiliation / Footnote Machinery
% ============================================================================



\newcommand\blfootnote[1]{%
  \begingroup
  \renewcommand\thefootnote{}\footnote{%
    \noindent 
    \hspace{-2em} 
    \parbox[t]{\columnwidth}{#1} 
  }%
  \addtocounter{footnote}{-1}%
  \endgroup
}

\renewcommand\addtolist[5][]{
  \begingroup
    \if\relax#3\relax\def\sep{}\else\def\sep{#5}\fi
    \let\protect\@unexpandable@protect
    \xdef#3{\expandafter{#3}\sep #4[#1]{#2}}%
  \endgroup
}

\newcommand\authorlist{}
\newcommand\authorformat[2][]{{\textbf{#2}$^{#1}$}}
\renewcommand\author[2][]{\addtolist[#1]{#2}{\authorlist}{\authorformat}{, }}

\newcommand\affiliationlist{}
\newcommand\affiliationformat[2][]{{\small\itshape $^{#1}$#2}}
\newcommand\affiliation[2][]{\addtolist[#1]{#2}{\affiliationlist}{\affiliationformat}{, }}

\newcommand{\contributionlist}{}
\newcommand{\contribution}[1]{\renewcommand{\contributionlist}{#1}}

\newcommand\metadatalist{}
\newcommand\metadataformat[2][]{{\small {\sffamily\bfseries #1:} #2}}
\newcommand\metadata[2][]{\addtolist[#1]{#2}{\metadatalist}{\metadataformat}{\par}}

% ============================================================================
% Title / Abstract / Date
% ============================================================================

\newcommand{\titlelist}{}
\renewcommand{\title}[1]{%
  \renewcommand{\titlelist}{{\LARGE\bfseries\sffamily #1}}%
}

\newcommand{\abstractlist}{}
\renewcommand{\abstract}[1]{%
  \long\def\abstractlist{{\color{fg} #1}}%
}

\newcommand{\titledate}{}
\renewcommand\date[1]{%
  \gdef\titledate{#1}%
}

\newcommand\correspondence[1]{\metadata[Correspondence]{#1}}
\newcommand{\email}[1]{\href{mailto:#1}{\texttt{#1}}}

\newcommand{\beginappendix}{\appendix{\huge\sffamily Appendix\par}}

% ============================================================================
% Title box
% ============================================================================

\RequirePackage[most]{tcolorbox}

\newcommand{\mymaketitle}{%

  \vspace*{-1.5cm} 

  \noindent
  \begin{minipage}[c]{0.75\linewidth}
    \includegraphics[height=1.0cm]{assets/ubc-logo-2018-crest-blue-cmyk.pdf}
    \hspace{0.25cm}
    \includegraphics[height=0.9cm]{assets/vector-institute-logo-vector.pdf}
    \hspace{0.2cm}
    \includegraphics[height=0.85cm]{assets/logo_transparent_background.png}
  \end{minipage}%
    \begin{minipage}[c]{0.25\linewidth}
      \raggedleft
      \raisebox{-0.8cm}{\sffamily\small\bfseries\color{accent}\titledate}
    \end{minipage}

  \vspace{0.3cm}

  \tcbset{
    enhanced,
    frame hidden,
    colback=bg,
    arc=8pt, 
    left=0.6cm,
    right=0.6cm,
    top=0.7cm,
    bottom=0.6cm,
    before skip=0pt
  }

  \begin{tcolorbox}
    \setlength{\parindent}{0pt}
    \setlength{\parskip}{0.5em}

    % --- Tile and Authors ---
    {\raggedright
      \setstretch{1.5}
      \nohyphens{\titlelist}\par
      \vskip 0.2cm
      \nohyphens{\authorlist}\par
      \vspace{-0.5em}  
      \affiliationlist\par
    }

    \vskip 0.2cm 

    % --- Abstract ---
    {\setstretch{1.25}\abstractlist\par}

    \vskip 0.4cm
    
    % --- Metadata ---
    {\raggedright
      \ifdefempty{\metadatalist}{}{%
        \metadatalist\par
      }
    }
  \end{tcolorbox}
  \FloatBarrier
}

% ============================================================================
% maketitle logic
% ============================================================================

\if@twocolumn
  \renewcommand{\maketitle}{
    \twocolumn[
      \mymaketitle
      \vskip 0.4cm
    ]
    \ifdefempty{\contributionlist}{}{%
      \blfootnote{\contributionlist}%
    }
  }
\else
  \renewcommand{\maketitle}{
    \mymaketitle
    \ifdefempty{\contributionlist}{}{%
      \blfootnote{\contributionlist}%
    }
  }
\fi