use leptos::ev::PointerEvent;
use leptos::prelude::*;
use leptos::svg;

/// How one number is written in a hover readout.
#[derive(Clone, Copy, PartialEq)]
pub enum Fmt {
    Hours,
    Billions,
    Loss,
    Percent,
    Rate,
}

fn format_value(fmt: Fmt, value: f64) -> String {
    match fmt {
        Fmt::Hours => format!("{value:.1} h"),
        Fmt::Billions => format!("{:.2}B tokens", value),
        Fmt::Loss => format!("{value:.2}"),
        Fmt::Percent => format!("{:.1}%", value * 100.0),
        Fmt::Rate => format_rate(value),
    }
}

fn format_rate(value: f64) -> String {
    if value <= 0.0 {
        return "0".to_string();
    }
    let mut exponent = value.log10().floor();
    let mut mantissa = (value / 10f64.powf(exponent) * 10.0).round() / 10.0;
    // Rounding can push the mantissa to 10.0, which is not a mantissa.
    if mantissa >= 10.0 {
        mantissa /= 10.0;
        exponent += 1.0;
    }
    let text = format!("{mantissa:.1}");
    let text = text.strip_suffix(".0").unwrap_or(&text);
    format!("{text}e{}", exponent as i32)
}

/// A chart axis, given as two (svg coordinate, data value) anchors read off the
/// tick markup. Every axis on these charts is linear, so two anchors define it.
#[derive(Clone, Copy)]
pub struct Axis {
    pub svg_a: f64,
    pub data_a: f64,
    pub svg_b: f64,
    pub data_b: f64,
}

impl Axis {
    fn value_at(&self, coordinate: f64) -> f64 {
        self.data_a
            + (coordinate - self.svg_a) * (self.data_b - self.data_a) / (self.svg_b - self.svg_a)
    }
}

#[derive(Clone)]
pub struct HoverSeries {
    label: &'static str,
    class: &'static str,
    points: Vec<(f64, f64)>,
}

impl HoverSeries {
    /// `points` is the same string the `<polyline>` renders, so the hover layer
    /// and the drawn curve can never drift apart.
    pub fn new(label: &'static str, class: &'static str, points: &str) -> Self {
        Self {
            label,
            class,
            points: parse_points(points),
        }
    }
}

fn parse_points(raw: &str) -> Vec<(f64, f64)> {
    raw.split_whitespace()
        .filter_map(|pair| {
            let (x, y) = pair.split_once(',')?;
            Some((x.parse().ok()?, y.parse().ok()?))
        })
        .collect()
}

#[derive(Clone)]
pub struct HoverSpec {
    pub view_w: f64,
    pub left: f64,
    pub right: f64,
    pub top: f64,
    pub bottom: f64,
    pub x_axis: Axis,
    pub y_axis: Axis,
    pub x_fmt: Fmt,
    pub y_fmt: Fmt,
    pub series: Vec<HoverSeries>,
}

/// Pointer readout for a chart. It snaps to the nearest sampled x, draws a
/// crosshair and a dot on every series, and prints the values at that sample.
/// Render it last inside the `<svg>`: its capture rect has to sit on top.
#[component]
pub fn ChartHover(spec: HoverSpec) -> impl IntoView {
    let capture: NodeRef<svg::Rect> = NodeRef::new();
    let active = RwSignal::new(None::<usize>);
    let spec = StoredValue::new(spec);

    let on_move = move |event: PointerEvent| {
        let Some(element) = capture.get_untracked() else {
            return;
        };
        let bounds = element.get_bounding_client_rect();
        if bounds.width() <= 0.0 {
            return;
        }
        let index = spec.with_value(|spec| {
            let fraction = (f64::from(event.client_x()) - bounds.left()) / bounds.width();
            let coordinate = spec.left + fraction * (spec.right - spec.left);
            spec.series
                .first()?
                .points
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    (a.0 - coordinate)
                        .abs()
                        .total_cmp(&(b.0 - coordinate).abs())
                })
                .map(|(index, _)| index)
        });
        if active.get_untracked() != index {
            active.set(index);
        }
    };

    let (left, top) = spec.with_value(|spec| (spec.left, spec.top));
    let (width, height) = spec.with_value(|spec| (spec.right - spec.left, spec.bottom - spec.top));

    view! {
        <g class="chart-hover" aria-hidden="true">
            {move || {
                let index = active.get()?;
                spec.with_value(|spec| {
                    let anchor = spec.series.first()?.points.get(index)?.0;
                    let mut fields = vec![format_value(spec.x_fmt, spec.x_axis.value_at(anchor))];
                    let dots = spec
                        .series
                        .iter()
                        .filter_map(|series| {
                            let (_, y) = *series.points.get(index)?;
                            let value = format_value(spec.y_fmt, spec.y_axis.value_at(y));
                            fields.push(match series.label {
                                "" => value,
                                label => format!("{label} {value}"),
                            });
                            Some(view! {
                                <circle
                                    class=format!("hover-dot {}", series.class)
                                    cx=anchor.to_string()
                                    cy=y.to_string()
                                    r="3.5"
                                ></circle>
                            })
                        })
                        .collect::<Vec<_>>();
                    let readout = fields.join(" · ");
                    // Rough advance width for 11px system sans, enough to decide
                    // whether the readout still fits to the right of the crosshair.
                    let text_width = readout.chars().count() as f64 * 5.6;
                    let flip = anchor + 10.0 + text_width > spec.view_w - 4.0;
                    let text_x = if flip { anchor - 10.0 } else { anchor + 10.0 };
                    let text_anchor = if flip { "end" } else { "start" };
                    Some(view! {
                        <line
                            class="hover-line"
                            x1=anchor.to_string()
                            x2=anchor.to_string()
                            y1=spec.top.to_string()
                            y2=spec.bottom.to_string()
                        ></line>
                        {dots}
                        <text
                            class="hover-readout"
                            x=text_x.to_string()
                            y=(spec.top + 10.0).to_string()
                            text-anchor=text_anchor
                        >
                            {readout}
                        </text>
                    })
                })
            }}
            <rect
                node_ref=capture
                class="hover-capture"
                x=left.to_string()
                y=top.to_string()
                width=width.to_string()
                height=height.to_string()
                on:pointermove=on_move
                on:pointerleave=move |_| active.set(None)
            ></rect>
        </g>
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rates_round_into_the_exponent() {
        assert_eq!(format_rate(1e-3), "1e-3");
        assert_eq!(format_rate(5e-4), "5e-4");
        assert_eq!(format_rate(1.5e-4), "1.5e-4");
        // Rounds to a mantissa of 10, which has to carry into the exponent.
        assert_eq!(format_rate(9.99e-5), "1e-4");
        assert_eq!(format_rate(0.0), "0");
    }

    #[test]
    fn points_parse_into_pairs() {
        assert_eq!(parse_points("1,2 3.5,4.5"), vec![(1.0, 2.0), (3.5, 4.5)]);
        assert_eq!(parse_points(""), Vec::new());
        // A malformed pair is dropped, the rest still parse.
        assert_eq!(parse_points("1,2 oops 3,4"), vec![(1.0, 2.0), (3.0, 4.0)]);
    }

    #[test]
    fn an_axis_maps_svg_coordinates_back_to_data() {
        // The y axis of the total loss chart: svg grows downward, data upward.
        let axis = Axis {
            svg_a: 212.6,
            data_a: 3.0,
            svg_b: 18.5,
            data_b: 6.0,
        };

        assert!((axis.value_at(212.6) - 3.0).abs() < 1e-9);
        assert!((axis.value_at(18.5) - 6.0).abs() < 1e-9);
        assert!((axis.value_at(115.55) - 4.5).abs() < 1e-9);
    }
}
