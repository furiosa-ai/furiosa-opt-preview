use proc_macro2::TokenStream;
use quote::quote;
use syn::{GenericParam, Generics, TypeParamBound};

pub(super) fn expansion(generics: &Generics, hidden: &syn::Ident) -> (TokenStream, TokenStream) {
    let axis = |bounds: &syn::punctuated::Punctuated<TypeParamBound, syn::Token![+]>| {
        bounds.iter().any(|bound| {
            matches!(bound, TypeParamBound::Trait(bound)
            if bound.path.segments.last().is_some_and(|segment| segment.ident == "AxisName"))
        })
    };
    let numeric = generics.params.iter().all(|param| match param {
        GenericParam::Lifetime(_) => true,
        GenericParam::Type(param) => {
            axis(&param.bounds)
                || generics
                    .where_clause
                    .iter()
                    .flat_map(|clause| &clause.predicates)
                    .any(|predicate| {
                        matches!(predicate, syn::WherePredicate::Type(predicate)
                    if matches!(&predicate.bounded_ty, syn::Type::Path(path)
                        if path.path.is_ident(&param.ident)) && axis(&predicate.bounds))
                    })
        }
        GenericParam::Const(param) => matches!(&param.ty, syn::Type::Path(path)
            if path.path.is_ident("usize")),
    });
    if numeric {
        let keys = generics.params.iter().filter_map(|param| match param {
            GenericParam::Lifetime(_) => None,
            GenericParam::Type(param) => {
                let ident = &param.ident;
                Some(quote!(<#ident as ::furiosa_opt_std::prelude::AxisName>::SIZE))
            }
            GenericParam::Const(param) => {
                let ident = &param.ident;
                Some(quote!(#ident))
            }
        });
        (
            quote!(),
            quote!(furiosa_opt_std::__private::Key::Numeric(&[#(#keys),*])),
        )
    } else {
        let arguments = generics.params.iter().filter_map(|param| match param {
            GenericParam::Lifetime(_) => None,
            GenericParam::Type(param) => Some(&param.ident),
            GenericParam::Const(param) => Some(&param.ident),
        });
        (
            quote!(#[furiosa_opt::instance_key = "type"]),
            quote!(furiosa_opt_std::__private::Key::Type(
                ::std::any::type_name_of_val(&self::#hidden::<#(#arguments),*>)
            )),
        )
    }
}
