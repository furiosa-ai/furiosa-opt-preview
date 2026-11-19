//! Macros for virtual ISA.

use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::quote;
use syn::spanned::Spanned;
use syn::visit_mut::{self, VisitMut};
use syn::{
    Attribute, Data, DeriveInput, Error, ExprForLoop, Item, Meta, Stmt, Type, Variant, parse_macro_input, parse_quote,
    parse_quote_spanned,
};

#[derive(Default)]
struct LoopHintMarkerInserter {
    errors: Vec<Error>,
}

/// A parsed loop hint attribute.
enum LoopDirective {
    Unroll,
    Bound(Box<syn::Expr>),
}

/// The rev of this macro's source tree, read at first expansion so no build script has to
/// stamp it (a build-script trigger on the macro would rebuild every dependent crate).
/// `FURIOSA_OPT_REV` overrides it (a pipeline building outside the origin tree). A published
/// snapshot carries its origin rev in the crate version (`0.5.1+g<sha>`), and that stamp wins
/// over git: a mirror checkout's own sha is the MIRROR commit, not the rev the driver was
/// built from. A registry checkout has neither and reads `unknown`, which the driver skips.
fn rev() -> &'static str {
    static REV: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    REV.get_or_init(|| {
        std::env::var("FURIOSA_OPT_REV")
            .ok()
            .or_else(|| {
                env!("CARGO_PKG_VERSION")
                    .split_once("+g")
                    .map(|(_, sha)| sha.to_owned())
                    .filter(|sha| !sha.is_empty())
            })
            .or_else(|| {
                let out = std::process::Command::new("git")
                    .args(["-C", env!("CARGO_MANIFEST_DIR"), "rev-parse", "--short=12", "HEAD"])
                    .output()
                    .ok()?;
                out.status
                    .success()
                    .then(|| String::from_utf8_lossy(&out.stdout).trim().to_owned())
            })
            .unwrap_or_else(|| "unknown".into())
    })
}

/// Fully unrolls a zero-based device loop with a positive compile-time end before scheduling.
/// Takes no arguments and cannot share a loop with [`bound`].
#[proc_macro_attribute]
pub fn unroll(attr: TokenStream, item: TokenStream) -> TokenStream {
    if !attr.is_empty() {
        return Error::new_spanned(TokenStream2::from(attr), "`unroll` takes no arguments")
            .to_compile_error()
            .into();
    }

    expand_loop_hint(item, LoopDirective::Unroll)
}

/// Declares a compile-time `usize` bound for a zero-based device loop; incompatible with [`unroll`].
/// `MAX` must be at least 1 and a compile-time end must equal it; a runtime end above `MAX` runs
/// the body `MAX` times on the device, and the CPU backend panics.
#[proc_macro_attribute]
pub fn bound(attr: TokenStream, item: TokenStream) -> TokenStream {
    if attr.is_empty() {
        return Error::new(Span::call_site(), BOUND_ARGUMENT).to_compile_error().into();
    }
    let max = parse_macro_input!(attr as syn::Expr);
    expand_loop_hint(item, LoopDirective::Bound(Box::new(max)))
}

/// The error for a `#[bound]` written without its one argument.
const BOUND_ARGUMENT: &str = "`bound` takes one argument, `MAX`: write `#[bound(MAX)]`";

fn expand_loop_hint(item: TokenStream, hint: LoopDirective) -> TokenStream {
    let mut loop_expr = parse_macro_input!(item as ExprForLoop);
    let mut errors = Vec::new();
    apply_loop_hint(&mut loop_expr, Some(hint), &mut errors);
    if !errors.is_empty() {
        let errors = errors.iter().map(Error::to_compile_error);
        return quote!(#(#errors)*).into();
    }
    quote!(#loop_expr).into()
}

impl VisitMut for LoopHintMarkerInserter {
    fn visit_expr_for_loop_mut(&mut self, loop_expr: &mut ExprForLoop) {
        // Consume loop-owned attributes before the generic visitor rejects misplaced ones.
        apply_loop_hint(loop_expr, None, &mut self.errors);
        visit_mut::visit_expr_for_loop_mut(self, loop_expr);
    }

    fn visit_attribute_mut(&mut self, attribute: &mut Attribute) {
        if LoopDirective::is_hint(attribute) {
            let path = attribute.path();
            let message = format!("`{}` goes on a `for` loop, and this is not one", quote!(#path));
            self.errors.push(Error::new_spanned(attribute, message));
        }
    }
}

/// Replaces the loop's sole hint, from attributes or `requested`, with a body-head marker.
fn apply_loop_hint(loop_expr: &mut ExprForLoop, requested: Option<LoopDirective>, errors: &mut Vec<Error>) {
    // The loop's single hint so far, with the span its marker reports at.
    let mut loop_hint = requested.map(|hint| (hint, Span::call_site()));
    loop_expr.attrs.retain(|attribute| {
        let Some(hint) = LoopDirective::parse(attribute) else {
            return true;
        };
        match (hint, &loop_hint) {
            (Ok(hint), Some((previous, _))) => {
                let message = if std::mem::discriminant(&hint) == std::mem::discriminant(previous) {
                    format!("duplicate `{}` attribute", hint.name())
                } else {
                    format!(
                        "`{}` and `{}` cannot annotate the same loop",
                        previous.name(),
                        hint.name()
                    )
                };
                errors.push(Error::new_spanned(attribute, message));
            }
            (Ok(hint), None) => loop_hint = Some((hint, attribute.span())),
            (Err(error), _) => errors.push(error),
        }
        false
    });
    if let Some((hint, span)) = loop_hint {
        hint.insert_marker(loop_expr, span);
    }
}

/// The loop index the `#[bound]` marker checks on the CPU, read through the loop's pattern; a `_`
/// is named first, and a pattern that binds no index by value yields `None`.
fn loop_index(pat: &mut syn::Pat) -> Option<syn::Expr> {
    match pat {
        // `i @ p` binds the whole index to `i`, whatever `p` matches. A `ref` index is refused: the
        // translator cannot read through it.
        syn::Pat::Ident(pat) if pat.by_ref.is_none() => {
            let ident = &pat.ident;
            Some(parse_quote!(#ident))
        }
        syn::Pat::Wild(_) => {
            let index = syn::Ident::new("__bound_index", Span::mixed_site());
            *pat = parse_quote!(#index);
            Some(parse_quote!(#index))
        }
        syn::Pat::Paren(paren) => loop_index(&mut paren.pat),
        _ => None,
    }
}

impl LoopDirective {
    /// Whether `attribute` is a loop hint, without parsing its arguments.
    fn is_hint(attribute: &Attribute) -> bool {
        let path = attribute.path();
        path.is_ident("unroll") || path.is_ident("bound")
    }

    /// Parses a loop hint attribute, or returns `None` for any other attribute.
    fn parse(attribute: &Attribute) -> Option<syn::Result<Self>> {
        if !Self::is_hint(attribute) {
            return None;
        }
        let path = attribute.path();
        Some(if path.is_ident("unroll") {
            match attribute.meta {
                Meta::Path(_) => Ok(Self::Unroll),
                _ => Err(Error::new_spanned(attribute, "`unroll` takes no arguments")),
            }
        } else if path.is_ident("bound") {
            match &attribute.meta {
                Meta::List(list) if !list.tokens.is_empty() => {
                    attribute.parse_args().map(|max| Self::Bound(Box::new(max)))
                }
                _ => Err(Error::new_spanned(attribute, BOUND_ARGUMENT)),
            }
        } else {
            unreachable!("`is_hint` admits a loop hint that `parse` does not handle")
        })
    }

    fn name(&self) -> &'static str {
        match self {
            Self::Unroll => "unroll",
            Self::Bound(_) => "bound",
        }
    }

    /// Inserts the call that carries this hint to the device translator at the head of the loop body,
    /// where the translator looks for it.
    fn insert_marker(self, loop_expr: &mut ExprForLoop, span: Span) {
        let marker: Stmt = match self {
            Self::Unroll => {
                parse_quote_spanned!(span=> ::furiosa_opt_std::__private::__loop_hint_unroll();)
            }
            // `const { }` makes rustc evaluate the bound, and unlike a const-generic argument it may
            // use generic parameters.
            Self::Bound(max) => match loop_index(&mut loop_expr.pat) {
                Some(index) => parse_quote_spanned!(span=>
                    ::furiosa_opt_std::__private::__loop_hint_bound(#index, const { #max });
                ),
                None => parse_quote_spanned!(span=>
                    ::core::compile_error!("`bound` needs its loop variable by value, as a name or `_`");
                ),
            },
        };
        loop_expr.body.stmts.insert(0, marker);
    }
}

#[proc_macro_attribute]
pub fn primitive(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attr_str = attr.to_string().trim_matches('"').to_owned();

    let mut item = parse_macro_input!(item as Item);
    if let Item::Enum(item_enum) = &mut item {
        for Variant { ident, attrs, .. } in &mut item_enum.variants {
            let variant_str = format!("{attr_str}::{ident}");
            attrs.push(parse_quote!(#[furiosa_opt::primitive = #variant_str]));
        }
    }

    let expanded = quote! {
        #[furiosa_opt::primitive = #attr_str]
        #item
    };
    expanded.into()
}

/// Derive macro for the DeviceSend trait.
///
/// Preserves owned, shared, and mutable field access when binding a struct.
///
/// # Compile-time Checks
///
/// Nested derived structs and arrays flatten their fields recursively.
///
#[proc_macro_derive(DeviceSend)]
pub fn device_send(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    // DeviceSend models a device-function argument: a tensor, or a struct/tuple of
    // them flattened positionally into launch inputs. Enums (variant-dependent layout)
    // and unions (no defined field set) have no positional flatten, so reject them.
    let fields = match &input.data {
        Data::Struct(data) => &data.fields,
        Data::Enum(_) | Data::Union(_) => {
            return syn::Error::new_spanned(name, "DeviceSend can only be derived for structs")
                .to_compile_error()
                .into();
        }
    };

    let tys: Vec<_> = fields.iter().map(|f| &f.ty).collect();
    let accessors: Vec<TokenStream2> = fields
        .iter()
        .enumerate()
        .map(|(i, f)| match &f.ident {
            Some(ident) => quote!(#ident),
            None => {
                let index = syn::Index::from(i);
                quote!(#index)
            }
        })
        .collect();

    let (_, ty_generics, _) = input.generics.split_for_impl();
    let mut generics = input.generics.clone();
    generics.make_where_clause().predicates.extend(
        tys.iter()
            .map(|ty| -> syn::WherePredicate { parse_quote!(#ty: ::furiosa_opt_std::runtime::DeviceSend) }),
    );
    let (generics, _, bounds) = generics.split_for_impl();
    quote! {
        impl #generics ::furiosa_opt_std::runtime::DeviceSend for #name #ty_generics #bounds {
            #[allow(unused_variables)]
            fn arguments<'__a>(
                &'__a self,
                access: ::furiosa_opt_std::runtime::Access,
                bind: &mut ::furiosa_opt_std::runtime::Bind<'__a, '_>,
            ) -> ::core::result::Result<(), ::furiosa_opt_std::Error> {
                #(::furiosa_opt_std::runtime::DeviceSend::arguments(&self.#accessors, access, bind)?;)*
                ::core::result::Result::Ok(())
            }
        }
    }
    .into()
}

/// Marks a function as a device entry point.
///
/// Defines `name` as a value whose type implements `DeviceFn`, so `launch(name, args)` runs it,
/// and `Loadable`, so `name.load(&device)?` returns a `Kernel` that launches without a lookup. `cargo <subcommand>` runs the Rust body on the
/// host; `cargo furiosa-opt <subcommand>` runs the compiled program on the NPU.
#[proc_macro_attribute]
pub fn device(attr: TokenStream, item: TokenStream) -> TokenStream {
    expand_device(&attr.to_string(), parse_macro_input!(item as Item)).into()
}

/// The `proc_macro` bridge exists only inside a real expansion, so the attribute's work takes
/// parsed input: unit tests drive it on `syn::parse_quote!` items.
fn expand_device(attr: &str, item: Item) -> TokenStream2 {
    let attr_int = |key: &str, default: usize| -> usize {
        attr.split(',')
            .filter_map(|kv| kv.split_once('='))
            .find(|(k, _)| k.trim() == key)
            .and_then(|(_, v)| v.trim().parse().ok())
            .unwrap_or(default)
    };
    let devices = attr_int("chip", 1) as u8;
    let pes = attr_int("pe", 8) as u8;
    let mut func = match item {
        Item::Fn(f) => f,
        other => {
            return syn::Error::new_spanned(other, "#[device] can only be applied to functions").to_compile_error();
        }
    };

    let mut loop_hints = LoopHintMarkerInserter::default();
    loop_hints.visit_block_mut(&mut func.block);
    if !loop_hints.errors.is_empty() {
        let errors = loop_hints.errors.iter().map(Error::to_compile_error);
        return quote!(#(#errors)*);
    }

    let vis = &func.vis;
    let name = &func.sig.ident;
    let name_str = name.to_string();
    let hidden = syn::Ident::new(&format!("__furiosa_opt_{name}"), name.span());
    let syn::Signature {
        inputs,
        output,
        generics,
        ..
    } = &func.sig;
    let is_context = |ty: &Type| {
        let Type::Reference(reference) = ty else {
            return false;
        };
        if reference.mutability.is_none() {
            return false;
        }
        let Type::Path(path) = reference.elem.as_ref() else {
            return false;
        };
        path.qself.is_none()
            && path
                .path
                .segments
                .last()
                .is_some_and(|segment| segment.ident == "Device")
    };
    let Some(syn::FnArg::Typed(context)) = inputs.first() else {
        return syn::Error::new_spanned(&func.sig, "the first #[device] parameter must be &mut Device")
            .to_compile_error();
    };
    if !is_context(&context.ty) {
        return syn::Error::new_spanned(&context.ty, "the first #[device] parameter must be &mut Device")
            .to_compile_error();
    }
    let params: Vec<_> = inputs
        .iter()
        .filter_map(|a| match a {
            syn::FnArg::Typed(pt) => Some(pt),
            _ => None,
        })
        .enumerate()
        .map(|(i, pt)| {
            let name = match pt.pat.as_ref() {
                syn::Pat::Ident(id) => id.ident.clone(),
                _ => syn::Ident::new(&format!("__arg_{i}"), proc_macro2::Span::call_site()),
            };
            let ty = &pt.ty;
            (name, quote!(#ty))
        })
        .collect();

    let types: Vec<_> = params.iter().map(|(_, ty)| ty).collect();

    // The leading `Device` names the runtime the function loads into; every other parameter,
    // lowering contexts included, binds positionally (contexts bind to nothing).
    let context = &params[0].0;
    let arg_names: Vec<&syn::Ident> = params.iter().skip(1).map(|(name, _)| name).collect();

    let return_ty = match output {
        syn::ReturnType::Default => quote!(()),
        syn::ReturnType::Type(_, ty) => quote!(#ty),
    };
    let tuple_type = if types.len() == 1 {
        quote!(#(#types)*)
    } else {
        quote!((#(#types),*))
    };
    let block = &func.block;

    let param_names: Vec<&syn::Ident> = params.iter().map(|(name, _)| name).collect();
    let body_destructure = if param_names.len() == 1 {
        quote!(#(#param_names)*)
    } else {
        quote!((#(#param_names),*))
    };

    let (impl_generics, _, where_clause) = generics.split_for_impl();
    let generic_keys: Vec<_> = generics
        .params
        .iter()
        .filter_map(|param| match param {
            syn::GenericParam::Lifetime(_) => None,
            syn::GenericParam::Type(param) => {
                let ident = &param.ident;
                Some(quote!(<#ident as ::furiosa_opt_std::prelude::AxisName>::SIZE))
            }
            syn::GenericParam::Const(param) => {
                let ident = &param.ident;
                Some(quote!(#ident))
            }
        })
        .collect();

    let path = quote!(concat!(module_path!(), "::", #name_str));

    // A registry entry per instantiation, the fn's path and its numeric axis/const values; `load`
    // returns a kernel whose type keeps the arguments that selected it.
    let loadable = {
        quote! {
            impl #impl_generics furiosa_opt_std::runtime::Loadable<#tuple_type> for #hidden #where_clause {
                const PATH: &'static str = #path;
                const KEY: &'static [usize] = &[#(#generic_keys),*];
                fn run<__FuriosaOptReturns: furiosa_opt_std::runtime::Returns<Self::Output>>(
                    __furiosa_opt_kernel: furiosa_opt_std::runtime::Kernel<Self, #tuple_type>,
                    #body_destructure: #tuple_type,
                    __furiosa_opt_returns: __FuriosaOptReturns,
                ) -> furiosa_opt_std::runtime::Submitted<
                    furiosa_opt_std::runtime::Launched<__FuriosaOptReturns::Value>,
                > {
                    furiosa_opt_std::__private::launch(
                        #context,
                        ::core::result::Result::Ok(__furiosa_opt_kernel),
                        (#(#arg_names,)*),
                        __furiosa_opt_returns,
                        |#context, (#(#arg_names,)*)| self::#hidden(#(#param_names),*),
                    )
                }
            }
        }
    };

    let rev = rev();
    quote! {
        #[furiosa_opt::rev = #rev]
        #[furiosa_opt::device = #attr]
        // `#[allow]` (not `#[expect]`): the hidden fn may or may not trigger
        // each of these lints depending on how the user defined the device
        // function, and `#[expect]` fails when the lint doesn't fire.
        #[allow(dead_code, unused, clippy::too_many_arguments)]
        fn #hidden #impl_generics (#inputs) #output #where_clause #block

        // Marker struct: the `__furiosa_opt_` prefix dodges a same-named module, and the braced (non-unit)
        // form keeps it out of the value namespace so it coexists with the hidden fn; npu `scan` strips it.
        #[allow(non_camel_case_types)]
        #[derive(Clone, Copy, Debug)]
        #vis struct #hidden {}

        // `#[allow]`: the const keeps the snake device-fn name, which trips `non_upper_case_globals`.
        #[allow(non_upper_case_globals)]
        #vis const #name: #hidden = #hidden {};

        impl #hidden {
            /// The device this function runs on, as `#[device(chip, pe)]` declares it.
            pub fn topology(&self) -> furiosa_opt_std::Topology {
                furiosa_opt_std::Topology { chips: #devices, pes: #pes }
            }

            /// The name the compiler uses for this function's registry entry.
            /// A caller that filters functions takes it from here rather than restating the path.
            #[doc(hidden)]
            pub fn path(&self) -> &'static str {
                #path
            }

        }

        impl #impl_generics furiosa_opt_std::runtime::DeviceFn<#tuple_type> for #hidden #where_clause {
            type Output = #return_ty;
            fn launch<__FuriosaOptReturns: furiosa_opt_std::runtime::Returns<Self::Output>>(
                self,
                #body_destructure: #tuple_type,
                __furiosa_opt_returns: __FuriosaOptReturns,
            ) -> furiosa_opt_std::runtime::Submitted<furiosa_opt_std::runtime::Launched<__FuriosaOptReturns::Value>> {
                // The host backend runs the Rust body instead of the loaded kernel.
                let __furiosa_opt_kernel = furiosa_opt_std::runtime::Loadable::<#tuple_type>::load(self, #context);
                furiosa_opt_std::__private::launch(
                    #context,
                    __furiosa_opt_kernel,
                    (#(#arg_names,)*),
                    __furiosa_opt_returns,
                    |#context, (#(#arg_names,)*)| self::#hidden(#(#param_names),*),
                )
            }
        }

        #loadable
    }
}

#[cfg(test)]
mod device_tests {
    use syn::{Item, parse_quote};

    use super::expand_device;

    #[test]
    fn accepts_only_mutable_context_references() {
        let rejects = |item: Item| expand_device("chip = 1", item).to_string().contains("compile_error");

        assert!(!rejects(parse_quote!(
            fn f(device: &'a mut furiosa_opt_std::Device) {}
        )));
        assert!(!rejects(parse_quote!(
            fn f(device: &mut Device<furiosa_opt_std::backend::Npu>) {}
        )));
        assert!(rejects(parse_quote!(
            fn f(device: &Device) {}
        )));
        assert!(rejects(parse_quote!(
            fn f(device: &mut TuContext<{ Tu::Main }>) {}
        )));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Turns the loop hints in `block` into markers as `#[device]` does, and returns the errors it
    /// reports.
    fn insert_device_markers(block: &mut syn::Block) -> Vec<String> {
        let mut inserter = LoopHintMarkerInserter::default();
        inserter.visit_block_mut(block);
        inserter.errors.iter().map(Error::to_string).collect()
    }

    /// Each hint becomes a marker call at the head of its loop body.
    #[test]
    fn hints_become_markers_at_the_head_of_the_loop_body() {
        let cases: [(syn::Block, Stmt); 2] = [
            (
                parse_quote!({
                    #[unroll]
                    for _ in 0..4 {
                        body();
                    }
                }),
                parse_quote!(::furiosa_opt_std::__private::__loop_hint_unroll();),
            ),
            (
                parse_quote!({
                    #[bound(N - 1)]
                    for _ in 0..n {
                        body();
                    }
                }),
                parse_quote!(::furiosa_opt_std::__private::__loop_hint_bound(__bound_index, const { N - 1 });),
            ),
        ];
        for (mut block, marker) in cases {
            assert!(insert_device_markers(&mut block).is_empty());
            let Stmt::Expr(syn::Expr::ForLoop(loop_expr), _) = &block.stmts[0] else {
                panic!("expected a `for` loop");
            };
            assert!(loop_expr.attrs.is_empty());
            let stmts = &loop_expr.body.stmts;
            assert_eq!(quote!(#(#stmts)*).to_string(), quote!(#marker body();).to_string());
        }
    }

    /// A loop takes at most one hint, whatever the kinds and their order.
    #[test]
    fn a_loop_takes_one_hint() {
        let cases: [(syn::Block, &str); 4] = [
            (
                parse_quote!({
                    #[unroll]
                    #[unroll]
                    for _ in 0..4 {}
                }),
                "duplicate `unroll` attribute",
            ),
            (
                parse_quote!({
                    #[bound(4)]
                    #[bound(8)]
                    for _ in 0..4 {}
                }),
                "duplicate `bound` attribute",
            ),
            (
                parse_quote!({
                    #[bound(4)]
                    #[unroll]
                    for _ in 0..4 {}
                }),
                "`bound` and `unroll` cannot annotate the same loop",
            ),
            (
                parse_quote!({
                    #[unroll]
                    #[bound(4)]
                    for _ in 0..4 {}
                }),
                "`unroll` and `bound` cannot annotate the same loop",
            ),
        ];
        for (mut block, message) in cases {
            assert_eq!(insert_device_markers(&mut block), [message]);
        }
    }

    /// A hint names itself when it is written on something other than a `for` loop.
    #[test]
    fn a_hint_goes_on_a_for_loop() {
        let cases: [(syn::Block, &str); 2] = [
            (
                parse_quote!({
                    #[unroll]
                    if ready() {}
                }),
                "`unroll` goes on a `for` loop, and this is not one",
            ),
            (
                parse_quote!({
                    #[bound(4)]
                    if ready() {}
                }),
                "`bound` goes on a `for` loop, and this is not one",
            ),
        ];
        for (mut block, message) in cases {
            assert_eq!(insert_device_markers(&mut block), [message]);
        }
    }

    /// A hint written with arguments it does not take is rejected.
    #[test]
    fn a_hint_takes_its_own_arguments() {
        let mut unroll = parse_quote!({
            #[unroll(2)]
            for _ in 0..4 {}
        });
        assert_eq!(insert_device_markers(&mut unroll), ["`unroll` takes no arguments"]);

        let missing: [syn::Block; 2] = [
            parse_quote!({
                #[bound]
                for _ in 0..4 {}
            }),
            parse_quote!({
                #[bound()]
                for _ in 0..4 {}
            }),
        ];
        for mut block in missing {
            assert_eq!(insert_device_markers(&mut block), [BOUND_ARGUMENT]);
        }

        // `syn` writes the parse error for two arguments, so check only that the bound is rejected
        // with one error, not the error's wording.
        let mut two = parse_quote!({
            #[bound(4, 8)]
            for _ in 0..4 {}
        });
        assert_eq!(insert_device_markers(&mut two).len(), 1);
    }

    /// When rustc expands `#[bound]` itself, outside a `#[device]` body, a remaining `#[unroll]`
    /// still conflicts with it.
    #[test]
    fn the_bound_attribute_rejects_a_remaining_unroll() {
        let mut loop_expr: ExprForLoop = parse_quote!(
            #[unroll]
            for _ in 0..4 {}
        );
        let mut errors = Vec::new();
        apply_loop_hint(&mut loop_expr, Some(LoopDirective::Bound(parse_quote!(4))), &mut errors);
        assert_eq!(errors.len(), 1);
        assert_eq!(
            errors[0].to_string(),
            "`bound` and `unroll` cannot annotate the same loop"
        );
    }
}
